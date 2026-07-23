from __future__ import annotations

from collections.abc import Mapping

import pytest

from gepaxmapelite.evaluation import (
    AmbiguityNoiseEvaluator,
    AmbiguityNoiseProbe,
    ProbeJudgment,
    ProbeStratum,
)
from gepaxmapelite.models import AmbiguityStrategy, NoiseStrategy
from gepaxmapelite.text_adapter import (
    TextCleanupAdapter,
    TextExample,
    TextJudgment,
)


def _probes() -> list[AmbiguityNoiseProbe]:
    return [
        AmbiguityNoiseProbe("clean", "clean input", ProbeStratum.CLEAN),
        AmbiguityNoiseProbe("amb", "ambiguous input", ProbeStratum.AMBIGUITY),
        AmbiguityNoiseProbe("noise", "noisy input", ProbeStratum.NOISE),
        AmbiguityNoiseProbe("mixed", "mixed input", ProbeStratum.MIXED),
        AmbiguityNoiseProbe(
            "meaning",
            "preserve this fact",
            ProbeStratum.MEANING_PRESERVATION,
        ),
    ]


def test_heldout_evaluator_derives_behavior_from_outputs_and_worst_stratum() -> None:
    scores = {"clean": 1.0, "amb": 0.8, "noise": 0.7, "mixed": 0.6, "meaning": 0.9}

    def execute(
        candidate: Mapping[str, str],
        probe: AmbiguityNoiseProbe,
        *,
        seed: int,
    ) -> str:
        assert candidate == {"task_prompt": "does not encode a bin"}
        assert seed >= 0
        return f"observed::{probe.probe_id}"

    def judge(
        probe: AmbiguityNoiseProbe,
        output: str,
        *,
        seed: int,
    ) -> ProbeJudgment:
        assert output == f"observed::{probe.probe_id}"
        assert seed >= 0
        ambiguity = {
            "amb": AmbiguityStrategy.ASSUME,
            "mixed": AmbiguityStrategy.CLARIFY,
        }.get(probe.probe_id)
        noise = {
            "noise": NoiseStrategy.FILTER,
            "mixed": NoiseStrategy.VERIFY,
        }.get(probe.probe_id)
        return ProbeJudgment(
            score=scores[probe.probe_id],
            ambiguity_strategy=ambiguity,
            noise_strategy=noise,
            feedback=f"feedback::{probe.probe_id}",
        )

    result = AmbiguityNoiseEvaluator(_probes(), execute, judge).evaluate(
        {"task_prompt": "does not encode a bin"},
        replicate_seed=11,
    )

    # Each policy has one observation at two different levels, so documented
    # conservative tie-breaking picks the higher ordinal behavior.
    assert result.descriptor.ambiguity is AmbiguityStrategy.CLARIFY
    assert result.descriptor.noise is NoiseStrategy.VERIFY
    assert result.fitness.quality == pytest.approx(0.6)
    assert result.fitness.eligible
    assert len(result.evidence) == 5
    assert result.feedback == tuple(f"feedback::{probe.probe_id}" for probe in _probes())


def test_probe_failure_becomes_an_ineligible_outer_evaluation() -> None:
    def execute(
        _candidate: Mapping[str, str],
        probe: AmbiguityNoiseProbe,
        *,
        seed: int,
    ) -> str:
        if probe.probe_id == "mixed":
            raise RuntimeError("model unavailable")
        return str(seed)

    def judge(
        probe: AmbiguityNoiseProbe,
        _output: str,
        *,
        seed: int,
    ) -> ProbeJudgment:
        return ProbeJudgment(
            score=1.0,
            ambiguity_strategy=(
                AmbiguityStrategy.BRANCH if probe.stratum is ProbeStratum.AMBIGUITY else None
            ),
            noise_strategy=(NoiseStrategy.EXTRACT if probe.stratum is ProbeStratum.NOISE else None),
        )

    result = AmbiguityNoiseEvaluator(_probes(), execute, judge).evaluate(
        {"task_prompt": "prompt"},
        replicate_seed=3,
    )
    assert not result.fitness.eligible
    assert result.fitness.mixed_score == 0.0
    assert "mixed:probe_error:mixed" in result.fitness.guardrail_violations


def test_text_cleanup_adapter_supplies_gepa_reflection_evidence() -> None:
    def execute(task_prompt: str, text: str) -> str:
        return f"{task_prompt}::{text.strip()}"

    def judge(example: TextExample, output: str) -> TextJudgment:
        assert example.text.strip() in output
        return TextJudgment(
            score=0.75,
            feedback="Keep the entity but remove the distractor.",
            objective_scores={"meaning": 1.0, "noise": 0.5},
        )

    adapter = TextCleanupAdapter(execute, judge)
    example = TextExample(" noisy ", example_id="one", reference="noisy")
    batch = adapter.evaluate(
        [example],
        {"task_prompt": "CLEAN"},
        capture_traces=True,
    )

    assert adapter.propose_new_texts is None
    assert batch.outputs == ["CLEAN::noisy"]
    assert batch.scores == [0.75]
    assert batch.objective_scores == [{"meaning": 1.0, "noise": 0.5}]
    reflective = adapter.make_reflective_dataset(
        {"task_prompt": "CLEAN"},
        batch,
        ["task_prompt"],
    )
    assert reflective["task_prompt"] == [
        {
            "input": " noisy ",
            "example_id": "one",
            "reference": "noisy",
            "output": "CLEAN::noisy",
            "score": 0.75,
            "feedback": "Keep the entity but remove the distractor.",
            "objective_scores": {"meaning": 1.0, "noise": 0.5},
            "metadata": {},
        }
    ]


def test_text_adapter_propagates_systemic_failures_unless_explicitly_recoverable() -> None:
    def systemic_failure(_task_prompt: str, _text: str) -> str:
        raise RuntimeError("provider authentication failed")

    def unused_judge(_example: TextExample, _output: str) -> TextJudgment:
        raise AssertionError("judge must not run")

    adapter = TextCleanupAdapter(systemic_failure, unused_judge)
    with pytest.raises(RuntimeError, match="authentication"):
        adapter.evaluate([TextExample("x")], {"task_prompt": "prompt"})

    def recoverable_failure(_task_prompt: str, _text: str) -> str:
        raise ValueError("one malformed example")

    recoverable = TextCleanupAdapter(
        recoverable_failure,
        unused_judge,
        recoverable_exceptions=(ValueError,),
    )
    batch = recoverable.evaluate(
        [TextExample("x")],
        {"task_prompt": "prompt"},
        capture_traces=True,
    )
    assert batch.scores == [0.0]
    assert "one malformed example" in batch.trajectories[0].judgment.feedback
