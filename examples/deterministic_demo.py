"""Offline demonstration of the complete nested archive boundary.

Run with:

    python examples/deterministic_demo.py

The scripted inner runner stands in for costly GEPA/model calls, while the
actual Promptbreeder, evolution engine, custom descriptor hash, archive,
persistence, and replacement logic all run unchanged.
"""

from __future__ import annotations

import argparse
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gepaxmapelite import (
    AmbiguityStrategy,
    BehaviorDescriptor,
    DownstreamEvaluation,
    EvolutionConfig,
    FitnessReport,
    MetaPromptGenome,
    MutationOperator,
    NoiseStrategy,
    PromptBreeder,
    evolve_meta_prompts,
)


def template(label: str) -> str:
    return (
        f"META::{label}\nInspect the current task prompt:\n<curr_param>\n"
        "Reflect on these execution traces:\n<side_info>\n"
        "Return exactly one improved task prompt."
    )


def genome(label: str) -> MetaPromptGenome:
    return MetaPromptGenome(
        reflection_template=template(label),
        mutation_prompt="Create a meaningfully different reflection strategy.",
    )


class ScriptedBreederLM:
    def __init__(self) -> None:
        self._outputs = iter((template("B"), template("C")))

    def __call__(self, _prompt: str, *, seed: int | None = None) -> str:
        assert seed is not None
        return next(self._outputs)


class ScriptedInnerRunner:
    """Replace this object with GEPAInnerRunner in a model-backed experiment."""

    def run(
        self,
        meta_prompt: MetaPromptGenome,
        *,
        replicate_seed: int,
        run_dir: str | Path,
    ) -> Mapping[str, Any]:
        label = meta_prompt.reflection_template.split("::", 1)[1].splitlines()[0]
        quality = {"A": 0.55, "B": 0.90, "C": 0.72}[label]
        return {
            "best_task_candidate": {"task_prompt": f"TASK::{label}"},
            "gepa_score": quality,
            "candidate_count": 4,
            "metric_calls": 20,
            "run_dir": str(run_dir),
        }


class ScriptedHeldoutEvaluator:
    def evaluate(
        self,
        task_candidate: Mapping[str, str],
        *,
        replicate_seed: int,
    ) -> DownstreamEvaluation:
        assert replicate_seed == 11
        label = task_candidate["task_prompt"].split("::", 1)[1]
        if label in {"A", "B"}:
            descriptor = BehaviorDescriptor(
                AmbiguityStrategy.STATE_ASSUMPTIONS,
                NoiseStrategy.FILTER,
            )
        else:
            descriptor = BehaviorDescriptor(
                AmbiguityStrategy.CLARIFY_OR_ABSTAIN,
                NoiseStrategy.VERIFY_REVISE,
            )
        quality = {"A": 0.55, "B": 0.90, "C": 0.72}[label]
        return DownstreamEvaluation(
            fitness=FitnessReport(quality, quality, quality, quality, quality),
            descriptor=descriptor,
            evidence=(),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or Path(tempfile.mkdtemp(prefix="gepaxmapelite-demo-"))

    result = evolve_meta_prompts(
        seeds=[genome("A")],
        breeder=PromptBreeder(
            ScriptedBreederLM(),
            rng=random.Random(3),
            operator_weights={MutationOperator.DIRECT: 1.0},
        ),
        inner_runner=ScriptedInnerRunner(),
        downstream_evaluator=ScriptedHeldoutEvaluator(),
        config=EvolutionConfig(
            offspring_count=2,
            run_dir=run_dir,
            master_seed=3,
            replicate_seeds=(11,),
        ),
    )

    print(f"archive: {result.archive.occupied_count}/16 bins; artifacts: {run_dir}")
    for key, elite in sorted(result.archive.elites().items()):
        print(f"bin {key}: quality={elite.quality:.2f}")
        print(f"  archived meta-prompt: {elite.genome.reflection_template.splitlines()[0]}")
        print(f"  best generated task prompt: {elite.best_task_candidate['task_prompt']}")
        print(
            f"  cell-representative task prompt: {elite.descriptor_task_candidate['task_prompt']}"
        )


if __name__ == "__main__":
    main()
