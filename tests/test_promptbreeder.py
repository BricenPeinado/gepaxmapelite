from __future__ import annotations

from collections.abc import Iterable

from gepaxmapelite.models import MetaPromptGenome
from gepaxmapelite.promptbreeder import (
    MutationOperator,
    PromptBreeder,
)

CURRENT_TEMPLATE = "Inspect <curr_param> against <side_info>, then propose a precise revision."
MUTATION_PROMPT = "Make the reflection strategy more robust."


class ScriptedLLM:
    def __init__(self, outputs: Iterable[str]) -> None:
        self._outputs = iter(outputs)
        self.calls: list[tuple[str, int | None]] = []

    def __call__(self, prompt: str, *, seed: int | None = None) -> str:
        self.calls.append((prompt, seed))
        try:
            return next(self._outputs)
        except StopIteration as exc:
            raise AssertionError("the breeder made an unexpected LLM call") from exc


class DrawSequence:
    """Small deterministic RNG used to exercise weighted interval boundaries."""

    def __init__(self, draws: Iterable[float]) -> None:
        self._draws = iter(draws)

    def random(self) -> float:
        try:
            return next(self._draws)
        except StopIteration as exc:
            raise AssertionError("the breeder requested an unexpected random draw") from exc


def parent_genome() -> MetaPromptGenome:
    return MetaPromptGenome(
        reflection_template=CURRENT_TEMPLATE,
        mutation_prompt=MUTATION_PROMPT,
    )


def test_direct_is_literal_m_plus_p_and_preserves_trace() -> None:
    revised = "Revised strategy for <curr_param> using <side_info>."
    llm = ScriptedLLM([revised])
    parent = parent_genome()

    result = PromptBreeder(llm).mutate(
        parent,
        operator=MutationOperator.DIRECT,
        lineage=("seed",),
        seed=41,
    )

    expected_prompt = f"{MUTATION_PROMPT}\n\n{CURRENT_TEMPLATE}"
    assert llm.calls == [(expected_prompt, 41)]
    assert result.raw_prompts == (expected_prompt,)
    assert result.raw_outputs == (revised,)
    assert result.call_seeds == (41,)
    assert result.operator is MutationOperator.DIRECT
    assert result.lineage == ("seed", MutationOperator.DIRECT.value)
    assert result.parent is parent
    assert result.genome.reflection_template == revised
    assert result.genome.mutation_prompt == MUTATION_PROMPT
    assert result.repaired_placeholders == ()


def test_first_order_hyper_is_h_plus_m_then_new_m_plus_p_with_sequential_seeds() -> None:
    hyper_prompt = "Invent a stronger mutation instruction."
    new_mutation = "Use counterexamples before selecting a revision."
    revised = "Counterexample-aware <curr_param> with evidence from <side_info>."
    llm = ScriptedLLM([new_mutation, revised])

    result = PromptBreeder(llm, hyper_prompt=hyper_prompt).mutate(
        parent_genome(),
        operator=MutationOperator.FIRST_ORDER_HYPER,
        seed=73,
    )

    first_call = f"{hyper_prompt}\n\n{MUTATION_PROMPT}"
    second_call = f"{new_mutation}\n\n{CURRENT_TEMPLATE}"
    assert llm.calls == [(first_call, 73), (second_call, 74)]
    assert result.raw_prompts == (first_call, second_call)
    assert result.raw_outputs == (new_mutation, revised)
    assert result.call_seeds == (73, 74)
    assert result.genome.mutation_prompt == new_mutation
    assert result.genome.reflection_template == revised
    assert result.operator is MutationOperator.FIRST_ORDER_HYPER


def test_missing_runtime_placeholders_are_repaired_without_altering_raw_output() -> None:
    raw_output = "Analyze the evidence and produce a safer revision."
    llm = ScriptedLLM([raw_output])

    result = PromptBreeder(llm).mutate(
        parent_genome(),
        operator=MutationOperator.DIRECT,
    )

    assert result.raw_outputs == (raw_output,)
    assert result.repaired_placeholders == ("<curr_param>", "<side_info>")
    assert result.genome.reflection_template.startswith(raw_output)
    assert result.genome.reflection_template.count("<curr_param>") == 1
    assert result.genome.reflection_template.count("<side_info>") == 1


def test_operator_weights_follow_deterministic_cumulative_intervals() -> None:
    rng = DrawSequence([0.0, 0.249, 0.25, 0.749, 0.75, 0.999])
    weights = {
        MutationOperator.DIRECT: 1.0,
        MutationOperator.FIRST_ORDER_HYPER: 2.0,
        MutationOperator.ZERO_ORDER_HYPER: 1.0,
    }
    breeder = PromptBreeder(lambda prompt: prompt, rng=rng, operator_weights=weights)

    chosen = [breeder.choose_operator() for _ in range(6)]

    assert chosen == [
        MutationOperator.DIRECT,
        MutationOperator.DIRECT,
        MutationOperator.FIRST_ORDER_HYPER,
        MutationOperator.FIRST_ORDER_HYPER,
        MutationOperator.ZERO_ORDER_HYPER,
        MutationOperator.ZERO_ORDER_HYPER,
    ]


def test_eda_prompt_uses_population_and_archive_templates() -> None:
    population_template = "Population method <curr_param> and <side_info>."
    archive_template = "Archive elite <curr_param> and <side_info>."
    population = [MetaPromptGenome(population_template, "Population mutation instruction.")]
    archive = [MetaPromptGenome(archive_template, "Archive mutation instruction.")]
    revised = "EDA synthesis for <curr_param> using <side_info>."
    llm = ScriptedLLM([revised])

    result = PromptBreeder(llm).mutate(
        parent_genome(),
        operator=MutationOperator.EDA,
        population=population,
        archive=archive,
        seed=5,
    )

    assert len(llm.calls) == 1
    prompt, call_seed = llm.calls[0]
    assert call_seed == 5
    assert CURRENT_TEMPLATE in prompt
    assert population_template in prompt
    assert archive_template in prompt
    assert "estimation-of-distribution" in prompt
    assert result.genome.reflection_template == revised
    assert result.genome.mutation_prompt == MUTATION_PROMPT


def test_eda_deduplicates_elites_already_present_in_population() -> None:
    shared = MetaPromptGenome(
        "Shared elite <curr_param> with <side_info>.",
        "shared mutation",
    )
    llm = ScriptedLLM(["Deduplicated <curr_param> with <side_info>."])

    PromptBreeder(llm).mutate(
        parent_genome(),
        operator=MutationOperator.EDA,
        population=(shared,),
        archive=(shared,),
    )

    assert llm.calls[0][0].count(shared.reflection_template) == 1


def test_automatic_selection_masks_lamarckian_without_successful_evidence() -> None:
    llm = ScriptedLLM(["Applicable mutation <curr_param> with <side_info>."])
    breeder = PromptBreeder(llm, rng=DrawSequence([0.999]))
    parent = parent_genome()

    result = breeder.mutate(
        parent,
        population=(parent,),
        archive=(parent,),
        population_scores=(0.5,),
        problem_description="Improve ambiguity handling.",
    )

    assert result.operator is MutationOperator.LINEAGE
    assert result.operator is not MutationOperator.LAMARCKIAN


def test_automatic_selection_masks_lamarckian_for_whitespace_only_evidence() -> None:
    llm = ScriptedLLM(["Applicable mutation <curr_param> with <side_info>."])
    breeder = PromptBreeder(
        llm,
        rng=DrawSequence([0.999]),
        operator_weights={
            MutationOperator.DIRECT: 1.0,
            MutationOperator.LAMARCKIAN: 1.0,
        },
    )

    result = breeder.mutate(
        parent_genome(),
        heldout_feedback=("   ",),
        working_outs="\n",
    )

    assert result.operator is MutationOperator.DIRECT


def test_automatic_selection_masks_ranked_eda_without_scored_population() -> None:
    llm = ScriptedLLM(["Applicable mutation <curr_param> with <side_info>."])
    breeder = PromptBreeder(
        llm,
        rng=DrawSequence([0.999]),
        operator_weights={
            MutationOperator.DIRECT: 1.0,
            MutationOperator.EDA_RANKED: 1.0,
        },
    )

    result = breeder.mutate(
        parent_genome(),
        archive=(parent_genome(),),
        population_scores=(),
    )

    assert result.operator is MutationOperator.DIRECT


def test_lamarckian_prompt_uses_successful_feedback_and_working_outs() -> None:
    feedback = "Held-out example improved after preserving the original negation."
    working = "Compare every number and negation before returning the revision."
    revised = "Feedback-aware <curr_param> with context from <side_info>."
    llm = ScriptedLLM([revised])

    result = PromptBreeder(llm).mutate(
        parent_genome(),
        operator=MutationOperator.LAMARCKIAN,
        heldout_feedback=(feedback,),
        working_outs=(working,),
        seed=19,
    )

    assert len(llm.calls) == 1
    prompt, call_seed = llm.calls[0]
    assert call_seed == 19
    assert feedback in prompt
    assert working in prompt
    assert CURRENT_TEMPLATE in prompt
    assert MUTATION_PROMPT in prompt
    assert "successful_heldout_feedback" in prompt
    assert "successful_working_outs" in prompt
    assert result.operator is MutationOperator.LAMARCKIAN
    assert result.genome.reflection_template == revised


def test_ranked_eda_orders_all_scored_references_from_weakest_to_strongest() -> None:
    weak = "Weak strategy <curr_param> with <side_info>."
    strong = "Strong strategy <curr_param> with <side_info>."
    unscored_archive = "Unscored archive <curr_param> with <side_info>."
    llm = ScriptedLLM(["Ranked synthesis <curr_param> with <side_info>."])

    PromptBreeder(llm).mutate(
        parent_genome(),
        operator=MutationOperator.EDA_RANKED,
        population=[
            MetaPromptGenome(strong, "strong mutation"),
            MetaPromptGenome(weak, "weak mutation"),
        ],
        population_scores=(0.9, 0.2),
        archive=[MetaPromptGenome(unscored_archive, "archive mutation")],
    )

    prompt = llm.calls[0][0]
    assert prompt.index(weak) < prompt.index(strong)
    assert unscored_archive not in prompt
    assert "ascending quality" in prompt


def test_lineage_operator_receives_chronological_ancestor_genotypes() -> None:
    root = MetaPromptGenome("Root <curr_param> with <side_info>.", "root mutation")
    parent = MetaPromptGenome("Parent <curr_param> with <side_info>.", "parent mutation")
    llm = ScriptedLLM(["Descendant <curr_param> with <side_info>."])

    PromptBreeder(llm).mutate(
        parent,
        operator=MutationOperator.LINEAGE,
        ancestor_templates=(root, parent),
    )

    prompt = llm.calls[0][0]
    assert prompt.index(root.reflection_template) < prompt.index(parent.reflection_template)
    assert "Ancestor 1 (oldest to newest)" in prompt
