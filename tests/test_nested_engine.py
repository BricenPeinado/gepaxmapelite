from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

import gepaxmapelite.gepa_runner as gepa_runner
from gepaxmapelite.archive import AdmissionReason, MapElitesArchive
from gepaxmapelite.engine import EvolutionConfig, evolve_meta_prompts
from gepaxmapelite.evaluation import (
    DownstreamEvaluation,
    ProbeEvidence,
    ProbeJudgment,
    ProbeStratum,
)
from gepaxmapelite.models import (
    AmbiguityStrategy,
    BehaviorDescriptor,
    FitnessReport,
    MetaPromptGenome,
    NoiseStrategy,
)
from gepaxmapelite.promptbreeder import MutationOperator, MutationResult, PromptBreeder

MUTATION_PROMPT = "Return one revised GEPA reflection template."


def _outer_template(label: str) -> str:
    return (
        f"OUTER::{label} inspect <curr_param> against <side_info>; "
        "return a revised task instruction."
    )


def _genome(label: str) -> MetaPromptGenome:
    return MetaPromptGenome(_outer_template(label), MUTATION_PROMPT)


def _fitness(quality: float) -> FitnessReport:
    return FitnessReport(
        clean_score=quality,
        ambiguity_score=quality,
        noise_score=quality,
        mixed_score=quality,
        meaning_preservation_score=quality,
    )


class ScriptedLLM:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[tuple[str, int | None]] = []

    def __call__(self, prompt: str, *, seed: int | None = None) -> str:
        self.calls.append((prompt, seed))
        if not self._outputs:
            raise AssertionError("unexpected PromptBreeder call")
        return self._outputs.pop(0)

    @property
    def remaining(self) -> int:
        return len(self._outputs)


class DeterministicInnerRunner:
    """Fake the inner GEPA boundary while recording its isolation contract."""

    def __init__(self) -> None:
        self.calls: list[tuple[MetaPromptGenome, int, Path]] = []

    def run(
        self,
        genome: MetaPromptGenome,
        *,
        replicate_seed: int,
        run_dir: str | Path,
    ) -> Mapping[str, Any]:
        path = Path(run_dir)
        self.calls.append((genome, replicate_seed, path))
        label = genome.reflection_template.split("::", 1)[1].split(" ", 1)[0]
        if label == "FAIL":
            raise RuntimeError("scripted inner evaluation failure")
        return {
            "best_task_candidate": {"task_prompt": f"TASK::{label}"},
            "gepa_score": {
                "A": 0.40,
                "B": 0.90,
                "C": 0.65,
                "D": 0.80,
            }[label],
            "candidate_count": 3,
            "metric_calls": 7,
            "run_dir": str(path),
        }


class FakeDownstreamEvaluator:
    """Assign behavior only from the generated task candidate it receives."""

    _OUTCOMES: ClassVar[dict[str, tuple[BehaviorDescriptor, float]]] = {
        "TASK::A": (
            BehaviorDescriptor(AmbiguityStrategy.BRANCH, NoiseStrategy.FILTER),
            0.40,
        ),
        "TASK::B": (
            BehaviorDescriptor(AmbiguityStrategy.BRANCH, NoiseStrategy.FILTER),
            0.90,
        ),
        "TASK::C": (
            BehaviorDescriptor(AmbiguityStrategy.CLARIFY, NoiseStrategy.VERIFY),
            0.65,
        ),
        "TASK::D": (
            BehaviorDescriptor(AmbiguityStrategy.CLARIFY, NoiseStrategy.VERIFY),
            0.80,
        ),
    }

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, str], int]] = []

    def evaluate(
        self,
        candidate: Mapping[str, str],
        *,
        replicate_seed: int,
    ) -> DownstreamEvaluation:
        materialized = dict(candidate)
        self.calls.append((materialized, replicate_seed))
        descriptor, quality = self._OUTCOMES[materialized["task_prompt"]]
        return DownstreamEvaluation(
            fitness=_fitness(quality),
            descriptor=descriptor,
            evidence=(),
        )


class TwoPolicyHasher:
    """A deliberately non-default binning hypothesis for the integration test."""

    version = "two-policy-test-v1"
    possible_keys = ("branch-filter", "clarify-verify")

    def __init__(self) -> None:
        self.calls: list[BehaviorDescriptor] = []

    def key(self, descriptor: BehaviorDescriptor) -> str:
        self.calls.append(descriptor)
        if descriptor == BehaviorDescriptor(
            AmbiguityStrategy.BRANCH,
            NoiseStrategy.FILTER,
        ):
            return "branch-filter"
        if descriptor == BehaviorDescriptor(
            AmbiguityStrategy.CLARIFY,
            NoiseStrategy.VERIFY,
        ):
            return "clarify-verify"
        raise AssertionError(f"unexpected behavior descriptor: {descriptor!r}")


def test_nested_evolution_preserves_outer_archive_boundary_and_reproducibility(
    tmp_path: Path,
) -> None:
    """Exercise PromptBreeder -> inner run -> evaluation -> MAP-Elites end to end."""

    # B replaces A in the same behavioral cell; C opens another cell; A is an
    # exact duplicate and must hit the cache; FAIL is recorded; D is still run
    # afterward and replaces C in its cell.
    llm = ScriptedLLM(
        [
            _outer_template("B"),
            _outer_template("C"),
            _outer_template("A"),
            _outer_template("FAIL"),
            _outer_template("D"),
        ]
    )
    breeder = PromptBreeder(
        llm,
        rng=random.Random(13),
        operator_weights={MutationOperator.DIRECT: 1.0},
    )
    inner = DeterministicInnerRunner()
    downstream = FakeDownstreamEvaluator()
    hasher = TwoPolicyHasher()
    run_dir = tmp_path / "nested-run"

    result = evolve_meta_prompts(
        seeds=[_genome("A")],
        breeder=breeder,
        inner_runner=inner,
        downstream_evaluator=downstream,
        config=EvolutionConfig(
            offspring_count=5,
            run_dir=run_dir,
            master_seed=73,
            replicate_seeds=(101, 202),
            continue_on_error=True,
            persist=True,
        ),
        descriptor_hasher=hasher,
    )

    assert llm.remaining == 0
    assert len(llm.calls) == 5
    assert all(seed is not None for _, seed in llm.calls)
    assert all(
        record.mutation is None or record.mutation.operator is MutationOperator.DIRECT
        for record in result.records
    )

    assert [record.proposal_id for record in result.records] == [0, 1, 2, 3, 5]
    assert result.unique_evaluations == 4
    assert result.cache_hits == 1
    duplicate = next(record for record in result.records if record.proposal_id == 3)
    assert duplicate.genome == _genome("A")
    assert duplicate.from_cache

    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.proposal_id == 4
    assert failure.stage == "offspring"
    assert failure.error_type == "RuntimeError"
    assert failure.message == "scripted inner evaluation failure"
    # Proposal 5 proves continue_on_error resumed evolution after proposal 4.
    assert any(record.proposal_id == 5 for record in result.records)

    replacement = next(record for record in result.records if record.proposal_id == 1)
    assert replacement.admission.reason is AdmissionReason.REPLACED_INCUMBENT
    assert replacement.admission.displaced is not None
    assert replacement.admission.displaced.genome == _genome("A")
    assert dict(replacement.admission.displaced.evaluation.best_inner_candidate) == {
        "task_prompt": "TASK::A"
    }

    assert set(result.archive) == {"branch-filter", "clarify-verify"}
    branch_elite = result.archive.elite("branch-filter")
    verify_elite = result.archive.elite("clarify-verify")
    assert branch_elite is not None
    assert verify_elite is not None
    assert branch_elite.genome == _genome("B")
    assert dict(branch_elite.evaluation.best_inner_candidate) == {"task_prompt": "TASK::B"}
    assert branch_elite.evaluation.best_inner_gepa_score == pytest.approx(0.90)
    assert branch_elite.evaluation.best_inner_candidate_count == 3
    assert branch_elite.evaluation.best_inner_run_id in branch_elite.evaluation.inner_run_ids
    assert verify_elite.genome == _genome("D")
    assert dict(verify_elite.evaluation.best_inner_candidate) == {"task_prompt": "TASK::D"}

    # The archive's selectable individual is always an outer reflection
    # meta-prompt; generated task prompts remain attached evaluation evidence.
    for elite in result.archive.elites().values():
        task_prompt = elite.evaluation.best_inner_candidate["task_prompt"]
        assert elite.evaluation.genome_id == elite.genome.genome_id
        assert elite.genome.reflection_template.startswith("OUTER::")
        assert task_prompt.startswith("TASK::")
        assert elite.genome.reflection_template != task_prompt

    # Every successfully evaluated genome receives the same replicate seeds,
    # each in a unique genome x replicate directory. The cached A proposal does
    # not launch another inner run; the failing proposal stops on its first seed.
    seeds_by_label: dict[str, list[int]] = defaultdict(list)
    for genome, replicate_seed, _ in inner.calls:
        label = genome.reflection_template.split("::", 1)[1].split(" ", 1)[0]
        seeds_by_label[label].append(replicate_seed)
    assert seeds_by_label == {
        "A": [101, 202],
        "B": [101, 202],
        "C": [101, 202],
        "FAIL": [101],
        "D": [101, 202],
    }
    passed_run_dirs = [path for _, _, path in inner.calls]
    assert len(passed_run_dirs) == len(set(passed_run_dirs))
    assert all(path.is_relative_to(run_dir / "inner_runs") for path in passed_run_dirs)
    assert not any("proposal-000003" in str(path) for path in passed_run_dirs)
    assert any("proposal-000005" in str(path) for path in passed_run_dirs)

    downstream_seeds: dict[str, list[int]] = defaultdict(list)
    for candidate, replicate_seed in downstream.calls:
        downstream_seeds[candidate["task_prompt"]].append(replicate_seed)
    assert downstream_seeds == {
        "TASK::A": [101, 202],
        "TASK::B": [101, 202],
        "TASK::C": [101, 202],
        "TASK::D": [101, 202],
    }
    assert result.total_metric_calls == 4 * 2 * 7

    # The injected hasher—not the default 4x4 mapping—owns the persisted bins.
    assert hasher.calls
    assert {(descriptor.ambiguity, descriptor.noise) for descriptor in hasher.calls} == {
        (AmbiguityStrategy.BRANCH, NoiseStrategy.FILTER),
        (AmbiguityStrategy.CLARIFY, NoiseStrategy.VERIFY),
    }
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["archive_individual"] == "gepa_reflection_meta_prompt"
    assert manifest["descriptor_hasher_version"] == hasher.version
    assert manifest["replicate_seeds"] == [101, 202]

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert [item["proposal_id"] for item in history["records"]] == [0, 1, 2, 3, 5]
    assert history["records"][3]["from_cache"] is True
    assert history["failures"][0]["proposal_id"] == 4
    assert history["metric_calls_consumed"] == 4 * 2 * 7
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["archive"] == result.archive.snapshot()
    assert checkpoint["history"] == history

    restored = MapElitesArchive.load(run_dir / "archive.json", hasher=hasher)
    assert restored.snapshot() == result.archive.snapshot()
    restored_branch = restored.elite("branch-filter")
    assert restored_branch is not None
    assert restored_branch.genome == _genome("B")
    assert dict(restored_branch.evaluation.best_inner_candidate) == {"task_prompt": "TASK::B"}


def test_engine_wires_successful_lamarckian_evidence_and_genotype_ancestry(
    tmp_path: Path,
) -> None:
    class CapturingBreeder(PromptBreeder):
        def __init__(self) -> None:
            super().__init__(
                ScriptedLLM([_outer_template("B")]),
                rng=random.Random(1),
                operator_weights={MutationOperator.DIRECT: 1.0},
            )
            self.contexts: list[dict[str, Any]] = []

        def mutate(self, parent: MetaPromptGenome, **kwargs: Any) -> MutationResult:
            self.contexts.append(dict(kwargs))
            return super().mutate(parent, operator=MutationOperator.DIRECT, **kwargs)

    class EvidenceEvaluator:
        def evaluate(
            self,
            candidate: Mapping[str, str],
            *,
            replicate_seed: int,
        ) -> DownstreamEvaluation:
            assert replicate_seed == 5
            label = candidate["task_prompt"].split("::", 1)[1]
            return DownstreamEvaluation(
                fitness=_fitness({"A": 0.4, "B": 0.9}[label]),
                descriptor=BehaviorDescriptor(
                    AmbiguityStrategy.ASSUME,
                    NoiseStrategy.FILTER,
                ),
                evidence=(
                    ProbeEvidence(
                        "good",
                        ProbeStratum.CLEAN,
                        "correct working out",
                        ProbeJudgment(0.9, feedback="successful lesson"),
                    ),
                    ProbeEvidence(
                        "bad",
                        ProbeStratum.NOISE,
                        "incorrect working out",
                        ProbeJudgment(0.2, feedback="failed lesson"),
                    ),
                    ProbeEvidence(
                        "unsafe",
                        ProbeStratum.MEANING_PRESERVATION,
                        "unsafe working out",
                        ProbeJudgment(
                            0.95,
                            feedback="unsafe lesson",
                            guardrail_violations=("invented fact",),
                        ),
                    ),
                    ProbeEvidence(
                        "blank",
                        ProbeStratum.CLEAN,
                        "  \n",
                        ProbeJudgment(0.95, feedback="   "),
                    ),
                ),
            )

    breeder = CapturingBreeder()
    evolve_meta_prompts(
        seeds=[_genome("A")],
        breeder=breeder,
        inner_runner=DeterministicInnerRunner(),
        downstream_evaluator=EvidenceEvaluator(),
        config=EvolutionConfig(
            offspring_count=1,
            run_dir=tmp_path / "context",
            replicate_seeds=(5,),
            lamarckian_success_threshold=0.8,
            persist=False,
        ),
    )

    context = breeder.contexts[0]
    assert context["heldout_feedback"] == ("successful lesson",)
    assert context["working_outs"] == ("correct working out",)
    assert context["ancestor_templates"] == (_genome("A"),)


def test_replicate_bin_matches_attached_candidate_and_uses_robust_scores(
    tmp_path: Path,
) -> None:
    class ReplicatedInner:
        def run(
            self,
            _genome: MetaPromptGenome,
            *,
            replicate_seed: int,
            run_dir: str | Path,
        ) -> Mapping[str, Any]:
            return {
                "best_task_candidate": {"task_prompt": f"TASK::{replicate_seed}"},
                "gepa_score": 0.7 if replicate_seed == 202 else 0.9,
                "candidate_count": 2,
                "metric_calls": 4,
                "run_dir": str(run_dir),
            }

    class ComplementaryEvaluator:
        def evaluate(
            self,
            candidate: Mapping[str, str],
            *,
            replicate_seed: int,
        ) -> DownstreamEvaluation:
            if replicate_seed == 101:
                fitness = FitnessReport(1.0, 0.0, 1.0, 1.0, 1.0)
                descriptor = BehaviorDescriptor(
                    AmbiguityStrategy.RESOLVE,
                    NoiseStrategy.SURFACE_CLEANUP,
                )
            else:
                fitness = FitnessReport(0.0, 1.0, 1.0, 1.0, 1.0)
                descriptor = BehaviorDescriptor(
                    AmbiguityStrategy.CLARIFY,
                    NoiseStrategy.VERIFY,
                )
            assert candidate["task_prompt"] == f"TASK::{replicate_seed}"
            return DownstreamEvaluation(fitness, descriptor, ())

    result = evolve_meta_prompts(
        seeds=[_genome("A")],
        breeder=PromptBreeder(lambda _prompt: _outer_template("unused")),
        inner_runner=ReplicatedInner(),
        downstream_evaluator=ComplementaryEvaluator(),
        config=EvolutionConfig(
            offspring_count=0,
            run_dir=tmp_path / "replicates",
            replicate_seeds=(101, 202),
            persist=False,
        ),
    )

    # Each replicate fails a different stratum, so worst-across-replicates
    # aggregation correctly remains zero instead of averaging to 0.5.
    assert result.archive.elites().keys() == {"15"}
    elite = result.archive.elite("15")
    assert elite is not None
    assert elite.quality == 0.0
    assert elite.descriptor == BehaviorDescriptor(
        AmbiguityStrategy.CLARIFY,
        NoiseStrategy.VERIFY,
    )
    # The globally strongest inner GEPA result and the task candidate that
    # represents the modal/tie-broken descriptor are preserved separately.
    assert elite.best_task_candidate["task_prompt"] == "TASK::101"
    assert elite.evaluation.best_inner_gepa_score == pytest.approx(0.9)
    assert elite.descriptor_task_candidate["task_prompt"] == "TASK::202"
    assert elite.evaluation.best_inner_run_id != (elite.evaluation.descriptor_representative_run_id)


def test_failed_downstream_evaluation_still_accounts_for_inner_cost(tmp_path: Path) -> None:
    class FailingEvaluator:
        def evaluate(
            self,
            _candidate: Mapping[str, str],
            *,
            replicate_seed: int,
        ) -> DownstreamEvaluation:
            raise RuntimeError(f"judge failed at seed {replicate_seed}")

    run_dir = tmp_path / "failed-cost"
    with pytest.raises(RuntimeError, match="no seed meta-prompt entered"):
        evolve_meta_prompts(
            seeds=[_genome("A")],
            breeder=PromptBreeder(lambda _prompt: _outer_template("unused")),
            inner_runner=DeterministicInnerRunner(),
            downstream_evaluator=FailingEvaluator(),
            config=EvolutionConfig(
                offspring_count=0,
                run_dir=run_dir,
                replicate_seeds=(5,),
                persist=True,
            ),
        )
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["history"]["metric_calls_consumed"] == 7


def test_failed_inner_run_reports_consumed_metric_calls_to_outer_ledger(tmp_path: Path) -> None:
    class CostedFailureRunner:
        def run(
            self,
            _genome: MetaPromptGenome,
            *,
            replicate_seed: int,
            run_dir: str | Path,
        ) -> Mapping[str, Any]:
            assert replicate_seed == 5
            raise gepa_runner.GEPAInnerRunFailure(
                "reflection failed",
                metric_calls=13,
                run_dir=Path(run_dir),
            )

    run_dir = tmp_path / "failed-inner-cost"
    with pytest.raises(RuntimeError, match="no seed meta-prompt entered"):
        evolve_meta_prompts(
            seeds=[_genome("A")],
            breeder=PromptBreeder(lambda _prompt: _outer_template("unused")),
            inner_runner=CostedFailureRunner(),
            downstream_evaluator=FakeDownstreamEvaluator(),
            config=EvolutionConfig(
                offspring_count=0,
                run_dir=run_dir,
                replicate_seeds=(5,),
                persist=True,
            ),
        )
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["history"]["metric_calls_consumed"] == 13


def test_gepa_wrapper_uses_meta_prompt_and_fresh_isolated_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimize_calls: list[dict[str, Any]] = []
    adapters: list[object] = []
    reflection_models: list[object] = []
    adapter_seeds: list[int] = []
    reflection_seeds: list[int] = []

    class Adapter:
        propose_new_texts = None

        def evaluate(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def make_reflective_dataset(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class ReflectionModel:
        def __call__(self, _prompt: str) -> str:
            return "proposal"

    def adapter_factory(*, seed: int) -> Adapter:
        adapter_seeds.append(seed)
        adapter = Adapter()
        adapters.append(adapter)
        return adapter

    def reflection_lm_factory(*, seed: int) -> ReflectionModel:
        reflection_seeds.append(seed)
        model = ReflectionModel()
        reflection_models.append(model)
        return model

    def optimize(**kwargs: Any) -> SimpleNamespace:
        optimize_calls.append(kwargs)
        kwargs["reflection_lm"]("reflection input")
        kwargs["callbacks"][0].on_proposal_end(None)
        return SimpleNamespace(
            best_idx=1,
            val_aggregate_scores=[0.25, 0.82],
            best_candidate={"task_prompt": "TASK::GEPA-BEST"},
            num_candidates=4,
            total_metric_calls=9,
        )

    monkeypatch.setattr(
        gepa_runner,
        "import_module",
        lambda name: SimpleNamespace(optimize=optimize) if name == "gepa" else None,
    )

    meta_prompt = _outer_template("WRAPPER")
    task_seed = {"task_prompt": "initial task prompt"}
    root = tmp_path / "gepa-runs"
    first = gepa_runner.run_gepa_inner(
        meta_prompt=meta_prompt,
        task_seed=task_seed,
        trainset=[{"text": "train"}],
        valset=[{"text": "validation"}],
        adapter_factory=adapter_factory,
        reflection_lm_factory=reflection_lm_factory,
        run_dir=root,
        seed=17,
        max_metric_calls=6,
    )
    second = gepa_runner.run_gepa_inner(
        meta_prompt=meta_prompt,
        task_seed=task_seed,
        trainset=[{"text": "train"}],
        valset=[{"text": "validation"}],
        adapter_factory=adapter_factory,
        reflection_lm_factory=reflection_lm_factory,
        run_dir=root,
        seed=17,
        max_metric_calls=6,
    )

    assert len(optimize_calls) == 2
    assert adapter_seeds == [17, 17]
    assert reflection_seeds == [17, 17]
    assert len(adapters) == len({id(adapter) for adapter in adapters}) == 2
    assert len(reflection_models) == len({id(model) for model in reflection_models}) == 2
    assert optimize_calls[0]["adapter"] is adapters[0]
    assert optimize_calls[1]["adapter"] is adapters[1]
    assert optimize_calls[0]["reflection_lm"].delegate is reflection_models[0]
    assert optimize_calls[1]["reflection_lm"].delegate is reflection_models[1]

    for call in optimize_calls:
        assert call["reflection_prompt_template"] == meta_prompt
        assert call["seed_candidate"] == task_seed
        assert call["seed_candidate"] is not task_seed
        assert call["seed"] == 17
        assert call["max_metric_calls"] == 6
        assert call["module_selector"] == "round_robin"
        assert call["use_merge"] is False
        assert call["cache_evaluation"] is False
        assert call["val_evaluation_policy"] == "full_eval"
        assert call["skip_perfect_score"] is False
        assert len(call["callbacks"]) == 1

    assert task_seed == {"task_prompt": "initial task prompt"}
    assert first["best_task_candidate"] == {"task_prompt": "TASK::GEPA-BEST"}
    assert first["gepa_score"] == pytest.approx(0.82)
    assert first["candidate_count"] == 4
    # GEPA checks its budget between iterations; the wrapper reports the actual
    # count instead of pretending the requested bound was exact.
    assert first["metric_calls"] == 9
    assert first["run_dir"] != second["run_dir"]
    for result in (first, second):
        isolated = Path(result["run_dir"])
        assert isolated.is_dir()
        assert isolated.parent == root


def test_gepa_wrapper_rejects_a_swallowed_reflection_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProviderFailure(RuntimeError):
        pass

    def reflection_lm(_prompt: str) -> str:
        raise ProviderFailure("provider unavailable")

    def optimize(**kwargs: Any) -> SimpleNamespace:
        # Reproduce GEPA 0.1.4's batch-safe behavior: it catches both attempts
        # and returns a normal-looking, seed-only result.
        for _ in range(2):
            with pytest.raises(ProviderFailure):
                kwargs["reflection_lm"]("reflection input")
        return SimpleNamespace(
            best_idx=0,
            val_aggregate_scores=[0.5],
            best_candidate={"task_prompt": "seed"},
            num_candidates=1,
            total_metric_calls=4,
        )

    monkeypatch.setattr(
        gepa_runner,
        "import_module",
        lambda name: SimpleNamespace(optimize=optimize) if name == "gepa" else None,
    )

    with pytest.raises(RuntimeError, match="before producing a valid proposal") as exc:
        gepa_runner.run_gepa_inner(
            meta_prompt=_outer_template("PROVIDER-FAILURE"),
            task_seed={"task_prompt": "seed"},
            trainset=[{"text": "train"}],
            valset=[{"text": "validation"}],
            adapter_factory=lambda: type(
                "Adapter",
                (),
                {
                    "propose_new_texts": None,
                    "evaluate": lambda *_args: None,
                    "make_reflective_dataset": lambda *_args: None,
                },
            )(),
            reflection_lm_factory=lambda: reflection_lm,
            run_dir=tmp_path,
            seed=3,
            max_metric_calls=4,
        )
    assert isinstance(exc.value.__cause__, ProviderFailure)
    assert exc.value.metric_calls == 4


def test_gepa_wrapper_rejects_a_partial_run_with_a_later_swallowed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProviderFailure(RuntimeError):
        pass

    calls = 0

    def reflection_lm(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "valid proposal"
        raise ProviderFailure("provider failed during a later iteration")

    def optimize(**kwargs: Any) -> SimpleNamespace:
        tracker = kwargs["callbacks"][0]
        tracker.on_proposal_start(None)
        kwargs["reflection_lm"]("first reflection")
        tracker.on_proposal_end(None)

        tracker.on_proposal_start(None)
        for _ in range(2):
            with pytest.raises(ProviderFailure):
                kwargs["reflection_lm"]("later reflection")
        return SimpleNamespace(
            best_idx=1,
            val_aggregate_scores=[0.5, 0.8],
            best_candidate={"task_prompt": "first improvement"},
            num_candidates=2,
            total_metric_calls=11,
        )

    monkeypatch.setattr(
        gepa_runner,
        "import_module",
        lambda name: SimpleNamespace(optimize=optimize) if name == "gepa" else None,
    )

    with pytest.raises(RuntimeError, match="completing every started proposal") as exc:
        gepa_runner.run_gepa_inner(
            meta_prompt=_outer_template("PARTIAL-PROVIDER-FAILURE"),
            task_seed={"task_prompt": "seed"},
            trainset=[{"text": "train"}],
            valset=[{"text": "validation"}],
            adapter_factory=lambda: type(
                "Adapter",
                (),
                {
                    "propose_new_texts": None,
                    "evaluate": lambda *_args: None,
                    "make_reflective_dataset": lambda *_args: None,
                },
            )(),
            reflection_lm_factory=lambda: reflection_lm,
            run_dir=tmp_path,
            seed=3,
            max_metric_calls=4,
        )
    assert isinstance(exc.value.__cause__, ProviderFailure)
    assert exc.value.metric_calls == 11


def test_model_name_reflection_factory_receives_provider_sampling_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[tuple[str, dict[str, Any]]] = []

    class SeededLM:
        def __init__(self, model: str, **kwargs: Any) -> None:
            constructed.append((model, kwargs))

        def __call__(self, _prompt: str) -> str:
            return "proposal"

    def optimize(**kwargs: Any) -> SimpleNamespace:
        kwargs["reflection_lm"]("reflection input")
        kwargs["callbacks"][0].on_proposal_end(None)
        return SimpleNamespace(
            best_idx=0,
            val_aggregate_scores=[0.6],
            best_candidate={"task_prompt": "evolved"},
            num_candidates=2,
            total_metric_calls=5,
        )

    def fake_import(name: str) -> Any:
        if name == "gepa":
            return SimpleNamespace(optimize=optimize)
        if name == "gepa.lm":
            return SimpleNamespace(LM=SeededLM)
        raise ImportError(name)

    monkeypatch.setattr(gepa_runner, "import_module", fake_import)

    gepa_runner.run_gepa_inner(
        meta_prompt=_outer_template("STRING-MODEL"),
        task_seed={"task_prompt": "seed"},
        trainset=[{"text": "train"}],
        valset=[{"text": "validation"}],
        adapter_factory=lambda: type(
            "Adapter",
            (),
            {
                "propose_new_texts": None,
                "evaluate": lambda *_args: None,
                "make_reflective_dataset": lambda *_args: None,
            },
        )(),
        reflection_lm_factory=lambda: "provider/model",
        run_dir=tmp_path,
        seed=23,
        max_metric_calls=4,
    )

    assert constructed == [("provider/model", {"seed": 23})]


def test_gepa_runner_rejects_a_budget_that_cannot_use_the_meta_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must exceed the valset size"):
        gepa_runner.run_gepa_inner(
            meta_prompt=_outer_template("BUDGET"),
            task_seed={"task_prompt": "seed"},
            trainset=[1, 2],
            valset=[1, 2],
            adapter_factory=lambda: object(),
            reflection_lm_factory=lambda: object(),
            run_dir=tmp_path,
            seed=1,
            max_metric_calls=2,
        )


def test_configured_runner_requires_fresh_factories_for_custom_loaders(
    tmp_path: Path,
) -> None:
    class MutableLoader:
        def __len__(self) -> int:
            return 1

        def all_ids(self) -> list[int]:
            return [0]

        def fetch(self, ids: list[int]) -> list[int]:
            return ids

    runner = gepa_runner.GEPAInnerRunner(
        task_seed={"task_prompt": "seed"},
        trainset=MutableLoader(),
        valset=MutableLoader(),
        adapter_factory=lambda: object(),
        reflection_lm_factory=lambda: object(),
        max_metric_calls=2,
    )
    with pytest.raises(TypeError, match="trainset_factory"):
        runner.run(_genome("LOADER"), replicate_seed=1, run_dir=tmp_path)
