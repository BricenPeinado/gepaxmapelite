"""Promptbreeder-inspired mutation of GEPA meta-prompts.

The outer Promptbreeder evolves :class:`MetaPromptGenome` instances.  A genome's
``reflection_template`` is the prompt being evolved (``P`` in the paper-style
notation used below), while ``mutation_prompt`` is the mutation instruction
(``M``).  This module deliberately knows nothing about the inner GEPA search;
it only produces the next meta-prompt genome and a complete mutation trace.

All language-model calls are routed through :class:`PromptBreeder`.  A callable
may accept just ``prompt`` or may additionally accept ``seed`` either by keyword
or position.  When a base seed is supplied, consecutive calls in a two-stage
hypermutation receive consecutive seeds.
"""

from __future__ import annotations

import inspect
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

from .models import MetaPromptGenome

CURRENT_PARAMETER_PLACEHOLDER = "<curr_param>"
SIDE_INFORMATION_PLACEHOLDER = "<side_info>"
REQUIRED_PLACEHOLDERS = (
    CURRENT_PARAMETER_PLACEHOLDER,
    SIDE_INFORMATION_PLACEHOLDER,
)


class LanguageModel(Protocol):
    """Structural protocol for the language model used by the breeder.

    ``PromptBreeder`` also supports prompt-only callables at runtime, making the
    optional seed an optimization rather than an integration requirement.
    """

    def __call__(self, prompt: str, *, seed: int | None = None) -> str:
        """Return a text completion for ``prompt``."""


LLMCallable: TypeAlias = Callable[..., str]


class MutationOperator(str, Enum):
    """Outer-loop mutation operators inspired by Promptbreeder."""

    DIRECT = "direct"
    FIRST_ORDER_HYPER = "first_order_hyper"
    ZERO_ORDER_HYPER = "zero_order_hyper"
    EDA = "eda"
    EDA_RANKED = "eda_ranked"
    LINEAGE = "lineage"
    LAMARCKIAN = "lamarckian"


DEFAULT_HYPER_PROMPT = """Create a new mutation instruction by improving the
mutation instruction below. The new instruction should encourage a meaningfully
different, useful revision strategy. Return only the new mutation instruction."""

DEFAULT_THINKING_STYLES = (
    "Work from invariants and failure modes before proposing a change.",
    "Generate several competing revisions, compare them, then synthesize one.",
    "Reason from concrete examples and counterexamples.",
    "Separate semantic preservation from stylistic improvement.",
)

_PRESERVATION_INSTRUCTION = (
    "The revised template must retain the exact runtime placeholders "
    f"{CURRENT_PARAMETER_PLACEHOLDER} and {SIDE_INFORMATION_PLACEHOLDER}. "
    "Return only the revised template."
)


@dataclass(frozen=True, slots=True)
class MutationResult:
    """The offspring genome and an auditable record of how it was produced.

    ``raw_prompts`` and ``raw_outputs`` are positionally paired and preserve
    the exact model traffic.  ``call_seeds`` records the seed actually supplied
    to each call; a value of ``None`` means either no seed was requested or the
    callable's signature did not accept one.
    """

    genome: MetaPromptGenome
    parent: MetaPromptGenome
    operator: MutationOperator
    lineage: tuple[str, ...]
    raw_prompts: tuple[str, ...]
    raw_outputs: tuple[str, ...]
    call_seeds: tuple[int | None, ...]
    repaired_placeholders: tuple[str, ...] = ()

    @property
    def offspring(self) -> MetaPromptGenome:
        """Alias that makes the parent/offspring relationship explicit."""

        return self.genome


GenomeOrTemplate: TypeAlias = MetaPromptGenome | str
OperatorKey: TypeAlias = MutationOperator | str


class PromptBreeder:
    """Generate mutated meta-prompt genomes with deterministic operator choice.

    Args:
        llm: Callable accepting a prompt, optionally with a ``seed`` argument.
        rng: Random source used only for operator and thinking-style selection.
            Supplying a seeded ``random.Random`` makes those choices repeatable.
        operator_weights: Optional partial mapping.  Operators omitted from a
            supplied mapping have zero weight.  With no mapping all operators
            are sampled uniformly.
        hyper_prompt: First-order instruction ``H`` used to transform ``M``.
        thinking_styles: Strategies available to zero-order hypermutation.
        problem_description: Default problem context for zero-order mutation.
    """

    def __init__(
        self,
        llm: LLMCallable,
        *,
        rng: random.Random | None = None,
        operator_weights: Mapping[OperatorKey, float] | None = None,
        hyper_prompt: str = DEFAULT_HYPER_PROMPT,
        thinking_styles: Sequence[str] = DEFAULT_THINKING_STYLES,
        problem_description: str = "",
    ) -> None:
        if not callable(llm):
            raise TypeError("llm must be callable")
        if rng is not None and not callable(getattr(rng, "random", None)):
            raise TypeError("rng must provide a random() method")

        self._llm = llm
        self._rng = rng if rng is not None else random.Random()
        self._operator_weights = _normalize_operator_weights(operator_weights)
        self._hyper_prompt = _nonempty_text(hyper_prompt, "hyper_prompt")
        self._thinking_styles = _text_tuple(
            thinking_styles, "thinking_styles", require_nonempty=True
        )
        self._problem_description = problem_description.strip()

    @property
    def operator_weights(self) -> Mapping[MutationOperator, float]:
        """Return a copy of the effective operator weights."""

        return dict(self._operator_weights)

    def choose_operator(self) -> MutationOperator:
        """Select an operator using a stable enum order and the configured RNG."""

        return self._choose_operator(frozenset(MutationOperator))

    def _choose_operator(
        self,
        available: frozenset[MutationOperator],
    ) -> MutationOperator:
        """Select among operators whose required mutation context is present."""

        unit_draw = float(self._rng.random())
        if not math.isfinite(unit_draw) or not 0.0 <= unit_draw < 1.0:
            raise ValueError("rng.random() must return a finite value in [0, 1)")

        total = sum(
            weight for operator, weight in self._operator_weights.items() if operator in available
        )
        if total <= 0.0:
            raise ValueError("no enabled mutation operator is applicable to the supplied context")
        target = unit_draw * total
        cumulative = 0.0
        final_enabled: MutationOperator | None = None
        for operator in MutationOperator:
            if operator not in available:
                continue
            weight = self._operator_weights[operator]
            if weight <= 0.0:
                continue
            final_enabled = operator
            cumulative += weight
            if target < cumulative:
                return operator

        # Floating-point summation can leave ``target`` infinitesimally above
        # the final cumulative boundary.  The last enabled operator is the
        # deterministic and mathematically correct fallback.
        if final_enabled is None:  # guarded by weight normalization
            raise RuntimeError("no mutation operator is enabled")
        return final_enabled

    def mutate(
        self,
        parent: MetaPromptGenome,
        *,
        operator: OperatorKey | None = None,
        population: Sequence[GenomeOrTemplate] = (),
        archive: Sequence[GenomeOrTemplate] = (),
        population_scores: Sequence[float] | None = None,
        problem_description: str | None = None,
        thinking_style: str | None = None,
        heldout_feedback: Sequence[str] | str = (),
        working_outs: Sequence[str] | str = (),
        lineage: Sequence[str] | str = (),
        ancestor_templates: Sequence[GenomeOrTemplate] = (),
        seed: int | None = None,
    ) -> MutationResult:
        """Mutate ``parent`` using one Promptbreeder operator.

        Contextual arguments are consumed only by their relevant operators:

        * ``population``/``archive`` feed EDA and provide a lineage fallback.
        * ``ancestor_templates`` supplies chronological genotypes to lineage mutation.
        * ``population_scores`` optionally ranks the population for EDA-Ranked.
        * ``problem_description``/``thinking_style`` feed zero-order mutation.
        * ``heldout_feedback``/``working_outs`` feed Lamarckian mutation.

        ``lineage`` contains caller-supplied ancestry labels.  The selected
        operator value is appended to it in the returned result.
        """

        if not isinstance(parent, MetaPromptGenome):
            raise TypeError("parent must be a MetaPromptGenome")
        current_prompt = _nonempty_text(parent.reflection_template, "parent.reflection_template")
        mutation_prompt = _nonempty_text(parent.mutation_prompt, "parent.mutation_prompt")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")

        if operator is None:
            available = set(MutationOperator)
            effective_description = (
                self._problem_description if problem_description is None else problem_description
            )
            if not isinstance(effective_description, str) or not effective_description.strip():
                available.discard(MutationOperator.ZERO_ORDER_HYPER)
            if not population and not archive:
                available.discard(MutationOperator.EDA)
                available.discard(MutationOperator.EDA_RANKED)
            if not population or population_scores is None:
                available.discard(MutationOperator.EDA_RANKED)
            if not ancestor_templates and not population and not archive:
                available.discard(MutationOperator.LINEAGE)
            if not _has_nonblank_text(heldout_feedback) and not _has_nonblank_text(working_outs):
                available.discard(MutationOperator.LAMARCKIAN)
            selected = self._choose_operator(frozenset(available))
        else:
            selected = _coerce_operator(operator)
        ancestry = _text_tuple(lineage, "lineage")
        raw_prompts: list[str] = []
        raw_outputs: list[str] = []
        call_seeds: list[int | None] = []

        def ask(prompt: str) -> str:
            call_seed = None if seed is None else seed + len(raw_prompts)
            output, supplied_seed = _invoke_llm(self._llm, prompt, call_seed)
            if not isinstance(output, str):
                raise TypeError("llm must return str")
            if not output.strip():
                raise ValueError("llm returned an empty completion")
            raw_prompts.append(prompt)
            raw_outputs.append(output)
            call_seeds.append(supplied_seed)
            return output.strip()

        next_mutation_prompt = mutation_prompt

        if selected is MutationOperator.DIRECT:
            # Paper notation: P' = LLM(M + P).
            candidate = ask(_application_prompt(mutation_prompt, current_prompt))

        elif selected is MutationOperator.FIRST_ORDER_HYPER:
            # Paper notation: M' = LLM(H + M), then P' = LLM(M' + P).
            next_mutation_prompt = ask(_first_order_prompt(self._hyper_prompt, mutation_prompt))
            candidate = ask(_application_prompt(next_mutation_prompt, current_prompt))

        elif selected is MutationOperator.ZERO_ORDER_HYPER:
            description = (
                self._problem_description
                if problem_description is None
                else problem_description.strip()
            )
            if not description:
                raise ValueError("zero-order hypermutation requires a problem_description")
            style = (
                _nonempty_text(thinking_style, "thinking_style")
                if thinking_style is not None
                else self._choose_thinking_style()
            )
            next_mutation_prompt = ask(_zero_order_prompt(description, style))
            candidate = ask(_application_prompt(next_mutation_prompt, current_prompt))

        elif selected is MutationOperator.EDA:
            references = _reference_templates(population, archive)
            if not references:
                raise ValueError("EDA mutation requires population or archive templates")
            candidate = ask(_eda_prompt(current_prompt, references, ranked=False))

        elif selected is MutationOperator.EDA_RANKED:
            if population_scores is None:
                raise ValueError("ranked EDA mutation requires population_scores")
            references = _ranked_reference_templates(population, archive, population_scores)
            if not references:
                raise ValueError("ranked EDA mutation requires population or archive templates")
            candidate = ask(_eda_prompt(current_prompt, references, ranked=True))

        elif selected is MutationOperator.LINEAGE:
            references = _chronological_reference_templates(ancestor_templates)
            if not references:
                references = _reference_templates(population, archive)
            if not references:
                raise ValueError("lineage mutation requires population or archive templates")
            candidate = ask(_lineage_prompt(current_prompt, references, ancestry))

        elif selected is MutationOperator.LAMARCKIAN:
            feedback = _text_tuple(heldout_feedback, "heldout_feedback")
            workings = _text_tuple(working_outs, "working_outs")
            if not feedback and not workings:
                raise ValueError("Lamarckian mutation requires heldout feedback or working outs")
            candidate = ask(
                _lamarckian_prompt(
                    current_prompt,
                    mutation_prompt,
                    feedback,
                    workings,
                )
            )

        else:  # pragma: no cover - Enum exhaustiveness guard
            raise AssertionError(f"unhandled mutation operator: {selected}")

        repaired_candidate, repaired = _repair_placeholders(candidate)
        offspring = MetaPromptGenome(
            reflection_template=repaired_candidate,
            mutation_prompt=next_mutation_prompt,
        )
        return MutationResult(
            genome=offspring,
            parent=parent,
            operator=selected,
            lineage=(*ancestry, selected.value),
            raw_prompts=tuple(raw_prompts),
            raw_outputs=tuple(raw_outputs),
            call_seeds=tuple(call_seeds),
            repaired_placeholders=repaired,
        )

    def _choose_thinking_style(self) -> str:
        unit_draw = float(self._rng.random())
        if not math.isfinite(unit_draw) or not 0.0 <= unit_draw < 1.0:
            raise ValueError("rng.random() must return a finite value in [0, 1)")
        index = min(int(unit_draw * len(self._thinking_styles)), len(self._thinking_styles) - 1)
        return self._thinking_styles[index]


def _normalize_operator_weights(
    weights: Mapping[OperatorKey, float] | None,
) -> dict[MutationOperator, float]:
    if weights is None:
        return {operator: 1.0 for operator in MutationOperator}
    if not isinstance(weights, Mapping):
        raise TypeError("operator_weights must be a mapping")

    normalized = {operator: 0.0 for operator in MutationOperator}
    seen: set[MutationOperator] = set()
    for key, value in weights.items():
        operator = _coerce_operator(key)
        if operator in seen:
            raise ValueError(f"duplicate weight for operator {operator.value!r}")
        seen.add(operator)
        if isinstance(value, bool):
            raise TypeError("operator weights must be numeric, not bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("operator weights must be numeric") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("operator weights must be finite and non-negative")
        normalized[operator] = numeric

    if sum(normalized.values()) <= 0.0:
        raise ValueError("at least one operator must have positive weight")
    return normalized


def _coerce_operator(value: OperatorKey) -> MutationOperator:
    if isinstance(value, MutationOperator):
        return value
    if not isinstance(value, str):
        raise TypeError("operator must be a MutationOperator or string")
    try:
        return MutationOperator(value.lower())
    except ValueError:
        try:
            return MutationOperator[value.upper()]
        except KeyError as exc:
            valid = ", ".join(operator.value for operator in MutationOperator)
            raise ValueError(f"unknown operator {value!r}; expected one of: {valid}") from exc


def _invoke_llm(
    llm: LLMCallable,
    prompt: str,
    seed: int | None,
) -> tuple[str, int | None]:
    """Call an LLM without masking a TypeError raised inside the callable."""

    if seed is None:
        return llm(prompt), None

    try:
        signature = inspect.signature(llm)
    except (TypeError, ValueError):
        # Without a signature there is no safe way to distinguish an unsupported
        # seed from a TypeError raised by the implementation itself.
        return llm(prompt), None

    try:
        signature.bind(prompt, seed=seed)
    except TypeError:
        try:
            signature.bind(prompt, seed)
        except TypeError:
            return llm(prompt), None
        return llm(prompt, seed), seed
    return llm(prompt, seed=seed), seed


def _application_prompt(mutation_prompt: str, current_prompt: str) -> str:
    # This intentionally remains a literal M + P composition. Placeholder
    # preservation is enforced after generation rather than by changing the
    # Promptbreeder operator itself.
    return f"{mutation_prompt.rstrip()}\n\n{current_prompt.rstrip()}"


def _first_order_prompt(hyper_prompt: str, mutation_prompt: str) -> str:
    # This intentionally remains a literal H + M composition.
    return f"{hyper_prompt.rstrip()}\n\n{mutation_prompt.rstrip()}"


def _zero_order_prompt(problem_description: str, thinking_style: str) -> str:
    return (
        "Create a novel mutation instruction for improving a meta-prompt. Do not "
        "rewrite the meta-prompt yet; return only the mutation instruction.\n\n"
        "<problem_description>\n"
        f"{problem_description.rstrip()}\n"
        "</problem_description>\n\n"
        "<thinking_style>\n"
        f"{thinking_style.rstrip()}\n"
        "</thinking_style>"
    )


def _eda_prompt(
    current_prompt: str,
    references: Sequence[str],
    *,
    ranked: bool,
) -> str:
    method = (
        "The scored examples are ranked in ascending quality, with the strongest "
        "examples last. Infer which recurring features correlate with quality, then "
        "continue the progression with a better template."
        if ranked
        else "Infer a useful distribution of strategies across the examples, then "
        "sample and synthesize a novel, better template."
    )
    return (
        "Use an estimation-of-distribution mutation over the reference "
        f"meta-prompts. {method}\n\n"
        f"{_format_reference_block(references)}\n\n"
        "<current_meta_prompt>\n"
        f"{current_prompt.rstrip()}\n"
        "</current_meta_prompt>\n\n"
        f"{_PRESERVATION_INSTRUCTION}"
    )


def _lineage_prompt(
    current_prompt: str,
    references: Sequence[str],
    lineage: Sequence[str],
) -> str:
    lineage_block = (
        "\n".join(f"{index + 1}. {step}" for index, step in enumerate(lineage))
        if lineage
        else "(no labeled ancestry supplied)"
    )
    return (
        "Create a descendant meta-prompt by studying the archive/population "
        "templates and the mutation lineage. Preserve successful inherited traits "
        "while introducing one coherent improvement.\n\n"
        f"{_format_reference_block(references)}\n\n"
        "<lineage>\n"
        f"{lineage_block}\n"
        "</lineage>\n\n"
        "<current_meta_prompt>\n"
        f"{current_prompt.rstrip()}\n"
        "</current_meta_prompt>\n\n"
        f"{_PRESERVATION_INSTRUCTION}"
    )


def _lamarckian_prompt(
    current_prompt: str,
    mutation_prompt: str,
    feedback: Sequence[str],
    working_outs: Sequence[str],
) -> str:
    return (
        "Revise the current meta-prompt using traits acquired from successful "
        "held-out evaluation feedback and working outs. Generalize the successful "
        "lessons; do not copy instance-specific answers or facts.\n\n"
        "<current_mutation_instruction>\n"
        f"{mutation_prompt.rstrip()}\n"
        "</current_mutation_instruction>\n\n"
        "<successful_heldout_feedback>\n"
        f"{_numbered_text(feedback)}\n"
        "</successful_heldout_feedback>\n\n"
        "<successful_working_outs>\n"
        f"{_numbered_text(working_outs)}\n"
        "</successful_working_outs>\n\n"
        "<current_meta_prompt>\n"
        f"{current_prompt.rstrip()}\n"
        "</current_meta_prompt>\n\n"
        f"{_PRESERVATION_INSTRUCTION}"
    )


def _reference_templates(
    population: Sequence[GenomeOrTemplate],
    archive: Sequence[GenomeOrTemplate],
) -> tuple[str, ...]:
    references: list[str] = []
    seen: set[str] = set()
    for label, collection in (("Population", population), ("Archive", archive)):
        label_index = 0
        for item in collection:
            template = _template_text(item)
            if template in seen:
                continue
            seen.add(template)
            label_index += 1
            references.append(f"{label} {label_index}:\n{template}")
    return tuple(references)


def _chronological_reference_templates(
    ancestors: Sequence[GenomeOrTemplate],
) -> tuple[str, ...]:
    return tuple(
        f"Ancestor {index + 1} (oldest to newest):\n{_template_text(item)}"
        for index, item in enumerate(ancestors)
    )


def _ranked_reference_templates(
    population: Sequence[GenomeOrTemplate],
    archive: Sequence[GenomeOrTemplate],
    population_scores: Sequence[float] | None,
) -> tuple[str, ...]:
    population_items = list(population)
    ranked_population: list[tuple[float | None, int, GenomeOrTemplate]]
    if population_scores is None:
        ranked_population = [(None, index, item) for index, item in enumerate(population_items)]
    else:
        scores = list(population_scores)
        if len(scores) != len(population_items):
            raise ValueError("population_scores must match population length")
        ranked_population = []
        for index, (item, input_score) in enumerate(zip(population_items, scores, strict=True)):
            if isinstance(input_score, bool):
                raise TypeError("population scores must be numeric, not bool")
            try:
                numeric = float(input_score)
            except (TypeError, ValueError) as exc:
                raise TypeError("population scores must be numeric") from exc
            if not math.isfinite(numeric):
                raise ValueError("population scores must be finite")
            ranked_population.append((numeric, index, item))
        ranked_population.sort(
            key=lambda entry: (
                entry[0] if entry[0] is not None else float("-inf"),
                entry[1],
            )
        )

    references: list[str] = []
    seen: set[str] = set()
    for display_score, _, item in ranked_population:
        template = _template_text(item)
        if template in seen:
            continue
        seen.add(template)
        rank = len(references) + 1
        score_text = "" if display_score is None else f" (score={display_score:g})"
        references.append(f"Ascending quality index {rank}{score_text}:\n{template}")
    if population_scores is None:
        offset = len(references)
        for item in archive:
            template = _template_text(item)
            if template in seen:
                continue
            seen.add(template)
            references.append(f"Archive elite {offset + 1}:\n{template}")
            offset += 1
    return tuple(references)


def _template_text(item: GenomeOrTemplate) -> str:
    if isinstance(item, MetaPromptGenome):
        return _nonempty_text(item.reflection_template, "reference reflection_template")
    if isinstance(item, str):
        return _nonempty_text(item, "reference template")
    raise TypeError("population and archive entries must be MetaPromptGenome or str")


def _format_reference_block(references: Sequence[str]) -> str:
    chunks = [
        f'<reference index="{index + 1}">\n{reference}\n</reference>'
        for index, reference in enumerate(references)
    ]
    return "<reference_meta_prompts>\n" + "\n\n".join(chunks) + "\n</reference_meta_prompts>"


def _numbered_text(items: Sequence[str]) -> str:
    if not items:
        return "(none supplied)"
    return "\n".join(f"{index + 1}. {item}" for index, item in enumerate(items))


def _repair_placeholders(template: str) -> tuple[str, tuple[str, ...]]:
    """Deterministically restore runtime placeholders omitted by the model."""

    repaired = template.strip()
    if not repaired:
        raise ValueError("cannot repair an empty reflection template")
    missing = tuple(token for token in REQUIRED_PLACEHOLDERS if token not in repaired)
    if not missing:
        return repaired, ()

    labels = {
        CURRENT_PARAMETER_PLACEHOLDER: "Current parameter",
        SIDE_INFORMATION_PLACEHOLDER: "Side information",
    }
    restoration = "\n".join(f"{labels[token]}: {token}" for token in missing)
    repaired = f"{repaired}\n\nRuntime inputs:\n{restoration}"
    return repaired, missing


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must not be empty")
    return stripped


def _text_tuple(
    values: Sequence[str] | str,
    name: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    candidates: tuple[str, ...]
    if isinstance(values, str):
        candidates = (values,)
    else:
        try:
            candidates = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{name} must be a sequence of strings") from exc

    result: list[str] = []
    for value in candidates:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain only strings")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must not contain empty strings")
        result.append(stripped)
    if require_nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    return tuple(result)


def _has_nonblank_text(values: Sequence[str] | str) -> bool:
    """Return whether contextual evidence contains a usable text item."""

    if isinstance(values, str):
        return bool(values.strip())
    return any(isinstance(value, str) and bool(value.strip()) for value in values)


__all__ = [
    "CURRENT_PARAMETER_PLACEHOLDER",
    "DEFAULT_HYPER_PROMPT",
    "DEFAULT_THINKING_STYLES",
    "REQUIRED_PLACEHOLDERS",
    "SIDE_INFORMATION_PLACEHOLDER",
    "LanguageModel",
    "MutationOperator",
    "MutationResult",
    "PromptBreeder",
]
