"""Nested Promptbreeder -> GEPA -> MAP-Elites evolution engine."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gepaxmapelite.archive import AdmissionDecision, MapElitesArchive
from gepaxmapelite.evaluation import DownstreamEvaluation, DownstreamEvaluator
from gepaxmapelite.hashing import DescriptorHasher
from gepaxmapelite.models import (
    AmbiguityStrategy,
    BehaviorDescriptor,
    FitnessReport,
    FrozenPromptCandidate,
    MetaPromptEvaluation,
    MetaPromptGenome,
    NoiseStrategy,
)
from gepaxmapelite.promptbreeder import MutationResult, PromptBreeder


class InnerRunner(Protocol):
    """Run a fresh inner optimizer for one outer meta-prompt and replicate."""

    def run(
        self,
        genome: MetaPromptGenome,
        *,
        replicate_seed: int,
        run_dir: str | Path,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class InnerRunEvidence:
    """Validated, immutable projection of an inner GEPA result."""

    best_task_candidate: FrozenPromptCandidate
    gepa_score: float
    candidate_count: int
    metric_calls: int
    run_dir: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> InnerRunEvidence:
        if not isinstance(value, Mapping):
            raise TypeError("inner runner must return a mapping")
        try:
            candidate = FrozenPromptCandidate(value["best_task_candidate"])
            score = float(value["gepa_score"])
            candidate_count = value["candidate_count"]
            metric_calls = value["metric_calls"]
            run_dir = value["run_dir"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("inner runner returned an invalid result") from exc
        if not math.isfinite(score):
            raise ValueError("inner GEPA score must be finite")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
            raise TypeError("inner candidate_count must be an integer")
        if candidate_count < 1:
            raise ValueError("inner GEPA run must produce at least one candidate")
        if isinstance(metric_calls, bool) or not isinstance(metric_calls, int):
            raise TypeError("inner metric_calls must be an integer")
        if metric_calls < 0:
            raise ValueError("inner metric_calls cannot be negative")
        if not isinstance(run_dir, str) or not run_dir:
            raise ValueError("inner run_dir must be a non-empty string")
        return cls(candidate, score, candidate_count, metric_calls, run_dir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_task_candidate": self.best_task_candidate.to_dict(),
            "gepa_score": self.gepa_score,
            "candidate_count": self.candidate_count,
            "metric_calls": self.metric_calls,
            "run_dir": self.run_dir,
        }


@dataclass(frozen=True, slots=True)
class ReplicateEvaluation:
    seed: int
    inner: InnerRunEvidence
    downstream: DownstreamEvaluation


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One evaluated outer individual and its archive admission result."""

    proposal_id: int
    genome: MetaPromptGenome
    descriptor: BehaviorDescriptor
    evaluation: MetaPromptEvaluation
    admission: AdmissionDecision
    replicates: tuple[ReplicateEvaluation, ...]
    parent_genome_id: str | None = None
    mutation: MutationResult | None = None
    from_cache: bool = False

    @property
    def lineage(self) -> tuple[str, ...]:
        return () if self.mutation is None else self.mutation.lineage


@dataclass(frozen=True, slots=True)
class EvolutionFailure:
    proposal_id: int
    stage: str
    error_type: str
    message: str
    parent_genome_id: str | None = None
    genome_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Outer-search budget and reproducibility settings."""

    offspring_count: int
    run_dir: Path | str
    master_seed: int = 0
    replicate_seeds: tuple[int, ...] = (0,)
    continue_on_error: bool = True
    persist: bool = True
    recent_population_size: int = 32
    lamarckian_success_threshold: float = 0.8
    problem_description: str = (
        "Create task prompts that remove irrelevant noise and handle ambiguity "
        "without changing facts, constraints, entities, or intent."
    )

    def __post_init__(self) -> None:
        for name in ("offspring_count", "master_seed", "recent_population_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.offspring_count < 0:
            raise ValueError("offspring_count cannot be negative")
        if self.master_seed < 0:
            raise ValueError("master_seed cannot be negative")
        if self.recent_population_size < 1:
            raise ValueError("recent_population_size must be positive")
        seeds = tuple(self.replicate_seeds)
        if not seeds:
            raise ValueError("replicate_seeds cannot be empty")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("replicate_seeds must contain non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("replicate_seeds cannot contain duplicates")
        object.__setattr__(self, "replicate_seeds", seeds)
        path = Path(self.run_dir).expanduser()
        object.__setattr__(self, "run_dir", path)
        if not isinstance(self.problem_description, str):
            raise TypeError("problem_description must be a string")
        threshold = self.lamarckian_success_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("lamarckian_success_threshold must be numeric")
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("lamarckian_success_threshold must be finite and in [0, 1]")
        object.__setattr__(self, "lamarckian_success_threshold", threshold)


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    archive: MapElitesArchive
    records: tuple[EvaluationRecord, ...]
    failures: tuple[EvolutionFailure, ...]
    unique_evaluations: int
    cache_hits: int
    metric_calls_consumed: int
    run_dir: Path

    @property
    def total_metric_calls(self) -> int:
        return self.metric_calls_consumed


class GenomeDigestCollisionError(RuntimeError):
    """A SHA-256 identity matched while the canonical genomes differed."""


class EvolutionPersistenceError(RuntimeError):
    """Persisting an otherwise successful in-memory transition failed."""


@dataclass(frozen=True, slots=True)
class _CachedEvaluation:
    genome: MetaPromptGenome
    descriptor: BehaviorDescriptor
    evaluation: MetaPromptEvaluation
    replicates: tuple[ReplicateEvaluation, ...]


def evolve_meta_prompts(
    *,
    seeds: Sequence[MetaPromptGenome],
    breeder: PromptBreeder,
    inner_runner: InnerRunner,
    downstream_evaluator: DownstreamEvaluator,
    config: EvolutionConfig,
    descriptor_hasher: DescriptorHasher | None = None,
) -> EvolutionResult:
    """Evolve and archive GEPA reflection meta-prompts.

    Seed genomes are evaluated first. Each subsequent proposal selects an
    archived meta-prompt, mutates it through Promptbreeder, gives the child its
    own isolated GEPA optimization, evaluates GEPA's best task prompt on held-out
    probes, and competes the *child meta-prompt* in the resulting behavior cell.
    """

    seed_genomes = tuple(seeds)
    if not seed_genomes:
        raise ValueError("at least one seed meta-prompt is required")
    if any(not isinstance(genome, MetaPromptGenome) for genome in seed_genomes):
        raise TypeError("all seeds must be MetaPromptGenome instances")
    if not isinstance(breeder, PromptBreeder):
        raise TypeError("breeder must be a PromptBreeder")
    if not callable(getattr(inner_runner, "run", None)):
        raise TypeError("inner_runner must provide run()")
    if not callable(getattr(downstream_evaluator, "evaluate", None)):
        raise TypeError("downstream_evaluator must provide evaluate()")
    if not isinstance(config, EvolutionConfig):
        raise TypeError("config must be an EvolutionConfig")

    archive = MapElitesArchive(descriptor_hasher)
    records: list[EvaluationRecord] = []
    failures: list[EvolutionFailure] = []
    cache: dict[str, _CachedEvaluation] = {}
    record_by_genome: dict[str, EvaluationRecord] = {}
    metric_call_ledger: list[int] = []
    cache_hits = 0

    run_root = config.run_dir
    assert isinstance(run_root, Path)
    (run_root / "inner_runs").mkdir(parents=True, exist_ok=True)
    if config.persist:
        _initialize_manifest(run_root, archive, config)

    def persist_or_raise(context: str) -> None:
        if not config.persist:
            return
        try:
            _persist_state(
                run_root,
                archive,
                records,
                failures,
                metric_calls_consumed=sum(metric_call_ledger),
            )
        except Exception as exc:
            raise EvolutionPersistenceError(f"failed to persist {context}") from exc

    def evaluate_and_insert(
        proposal_id: int,
        genome: MetaPromptGenome,
        *,
        parent_genome_id: str | None = None,
        mutation: MutationResult | None = None,
    ) -> EvaluationRecord:
        nonlocal cache_hits
        cached = cache.get(genome.genome_id)
        if cached is not None:
            if cached.genome != genome:
                raise GenomeDigestCollisionError(
                    "two unequal meta-prompt genomes produced the same identity digest"
                )
            descriptor = cached.descriptor
            evaluation = cached.evaluation
            replicates = cached.replicates
            from_cache = True
            cache_hits += 1
        else:
            replicates = _evaluate_replicates(
                proposal_id,
                genome,
                inner_runner,
                downstream_evaluator,
                config,
                metric_call_ledger,
            )
            descriptor, evaluation = _aggregate_replicates(genome, replicates)
            cache[genome.genome_id] = _CachedEvaluation(
                genome=genome,
                descriptor=descriptor,
                evaluation=evaluation,
                replicates=replicates,
            )
            from_cache = False

        admission = archive.insert(genome, evaluation, descriptor)
        record = EvaluationRecord(
            proposal_id=proposal_id,
            genome=genome,
            descriptor=descriptor,
            evaluation=evaluation,
            admission=admission,
            replicates=replicates,
            parent_genome_id=parent_genome_id,
            mutation=mutation,
            from_cache=from_cache,
        )
        records.append(record)
        if admission.accepted:
            # Parent context must track the record that actually owns the
            # active archive cell. A cached duplicate can be rejected as a tie
            # and must not replace the incumbent's Promptbreeder lineage.
            record_by_genome[genome.genome_id] = record
        persist_or_raise(f"proposal {proposal_id}")
        return record

    next_proposal_id = 0
    for genome in seed_genomes:
        try:
            evaluate_and_insert(next_proposal_id, genome)
        except EvolutionPersistenceError:
            raise
        except Exception as exc:
            failure = _failure(next_proposal_id, "seed_evaluation", exc, genome=genome)
            failures.append(failure)
            persist_or_raise(f"seed failure {next_proposal_id}")
            if not config.continue_on_error:
                raise
        next_proposal_id += 1

    if not archive:
        raise RuntimeError(
            "no seed meta-prompt entered the archive; inspect failures and guardrail violations"
        )

    for _ in range(config.offspring_count):
        proposal_id = next_proposal_id
        next_proposal_id += 1
        parent_rng = random.Random(_derived_seed(config.master_seed, proposal_id, "parent"))
        parent_elite = archive.select_uniform(parent_rng)
        parent_record = record_by_genome[parent_elite.genome.genome_id]
        recent_records = _recent_unique_records(records, config.recent_population_size)
        active_records = tuple(
            record_by_genome[archive.elites()[key].genome.genome_id] for key in archive
        )
        population_records = _merge_unique_records(recent_records, active_records)
        population = tuple(record.genome for record in population_records)
        population_scores = tuple(record.evaluation.quality for record in population_records)
        archive_genomes = tuple(archive.elites()[key].genome for key in archive)
        successful_evidence = tuple(
            evidence
            for replicate in parent_record.replicates
            for evidence in replicate.downstream.evidence
            if evidence.judgment.score >= config.lamarckian_success_threshold
            and not evidence.judgment.guardrail_violations
        )
        feedback = tuple(
            evidence.judgment.feedback.strip()
            for evidence in successful_evidence
            if evidence.judgment.feedback.strip()
        )
        working_outs = tuple(
            evidence.output.strip() for evidence in successful_evidence if evidence.output.strip()
        )
        ancestor_templates = _ancestral_genomes(parent_record, record_by_genome)

        try:
            mutation = breeder.mutate(
                parent_elite.genome,
                population=population,
                archive=archive_genomes,
                population_scores=population_scores,
                problem_description=config.problem_description or None,
                heldout_feedback=feedback,
                working_outs=working_outs,
                lineage=parent_record.lineage,
                ancestor_templates=ancestor_templates,
                seed=_derived_seed(config.master_seed, proposal_id, "mutation"),
            )
            evaluate_and_insert(
                proposal_id,
                mutation.genome,
                parent_genome_id=parent_elite.genome.genome_id,
                mutation=mutation,
            )
        except EvolutionPersistenceError:
            raise
        except Exception as exc:
            failed_genome: MetaPromptGenome | None = (
                mutation.genome if "mutation" in locals() else None
            )
            failure = _failure(
                proposal_id,
                "offspring",
                exc,
                parent_genome_id=parent_elite.genome.genome_id,
                genome=failed_genome,
            )
            failures.append(failure)
            persist_or_raise(f"offspring failure {proposal_id}")
            if not config.continue_on_error:
                raise
        finally:
            # Avoid accidentally associating a previous loop's child with a
            # failure that happened before this loop produced a mutation.
            if "mutation" in locals():
                del mutation

    return EvolutionResult(
        archive=archive,
        records=tuple(records),
        failures=tuple(failures),
        unique_evaluations=len(cache),
        cache_hits=cache_hits,
        metric_calls_consumed=sum(metric_call_ledger),
        run_dir=run_root,
    )


def _evaluate_replicates(
    proposal_id: int,
    genome: MetaPromptGenome,
    inner_runner: InnerRunner,
    downstream_evaluator: DownstreamEvaluator,
    config: EvolutionConfig,
    metric_call_ledger: list[int],
) -> tuple[ReplicateEvaluation, ...]:
    run_root = config.run_dir
    assert isinstance(run_root, Path)
    results: list[ReplicateEvaluation] = []
    for replicate_index, seed in enumerate(config.replicate_seeds):
        replicate_dir = (
            run_root
            / "inner_runs"
            / f"proposal-{proposal_id:06d}"
            / f"replicate-{replicate_index:03d}"
        )
        try:
            raw_inner = inner_runner.run(
                genome,
                replicate_seed=seed,
                run_dir=replicate_dir,
            )
        except Exception as exc:
            consumed = getattr(exc, "metric_calls", None)
            if isinstance(consumed, int) and not isinstance(consumed, bool) and consumed >= 0:
                metric_call_ledger.append(consumed)
            raise
        inner = InnerRunEvidence.from_mapping(raw_inner)
        metric_call_ledger.append(inner.metric_calls)
        downstream = downstream_evaluator.evaluate(
            inner.best_task_candidate,
            replicate_seed=seed,
        )
        if not isinstance(downstream, DownstreamEvaluation):
            raise TypeError("downstream evaluator must return DownstreamEvaluation")
        results.append(ReplicateEvaluation(seed, inner, downstream))
    return tuple(results)


def _aggregate_replicates(
    genome: MetaPromptGenome,
    replicates: Sequence[ReplicateEvaluation],
) -> tuple[BehaviorDescriptor, MetaPromptEvaluation]:
    if not replicates:
        raise ValueError("cannot aggregate zero replicates")
    descriptions = Counter(
        (
            replicate.downstream.descriptor.ambiguity,
            replicate.downstream.descriptor.noise,
        )
        for replicate in replicates
    )
    largest_group = max(descriptions.values())
    tied_descriptors = tuple(pair for pair, count in descriptions.items() if count == largest_group)

    def descriptor_rank(
        pair: tuple[AmbiguityStrategy, NoiseStrategy],
    ) -> tuple[int, float, int, int]:
        matching = [
            replicate
            for replicate in replicates
            if (
                replicate.downstream.descriptor.ambiguity,
                replicate.downstream.descriptor.noise,
            )
            == pair
        ]
        return (
            max(int(replicate.downstream.fitness.eligible) for replicate in matching),
            max(replicate.downstream.fitness.quality for replicate in matching),
            int(pair[0]),
            int(pair[1]),
        )

    ambiguity, noise = max(
        tied_descriptors,
        key=descriptor_rank,
    )
    descriptor = BehaviorDescriptor(
        ambiguity=AmbiguityStrategy(ambiguity),
        noise=NoiseStrategy(noise),
    )

    reports = [replicate.downstream.fitness for replicate in replicates]
    fitness = FitnessReport(
        clean_score=_minimum(report.clean_score for report in reports),
        ambiguity_score=_minimum(report.ambiguity_score for report in reports),
        noise_score=_minimum(report.noise_score for report in reports),
        mixed_score=_minimum(report.mixed_score for report in reports),
        meaning_preservation_score=_minimum(
            report.meaning_preservation_score for report in reports
        ),
        guardrail_violations=tuple(
            dict.fromkeys(
                violation for report in reports for violation in report.guardrail_violations
            )
        ),
    )
    descriptor_replicates = tuple(
        replicate for replicate in replicates if replicate.downstream.descriptor == descriptor
    )
    representative = sorted(
        descriptor_replicates,
        key=lambda replicate: (
            -int(replicate.downstream.fitness.eligible),
            -replicate.downstream.fitness.quality,
            -replicate.inner.gepa_score,
            replicate.inner.metric_calls,
            json.dumps(replicate.inner.best_task_candidate.to_dict(), sort_keys=True),
        ),
    )[0]
    best_inner = sorted(
        replicates,
        key=lambda replicate: (
            -replicate.inner.gepa_score,
            -int(replicate.downstream.fitness.eligible),
            -replicate.downstream.fitness.quality,
            replicate.inner.metric_calls,
            json.dumps(replicate.inner.best_task_candidate.to_dict(), sort_keys=True),
        ),
    )[0]
    qualities = [report.quality for report in reports]
    variance = statistics.pvariance(qualities) if len(qualities) > 1 else 0.0
    evaluation = MetaPromptEvaluation(
        genome_id=genome.genome_id,
        best_inner_candidate=best_inner.inner.best_task_candidate,
        fitness=fitness,
        total_cost=float(sum(replicate.inner.metric_calls for replicate in replicates)),
        quality_variance=variance,
        inner_run_ids=tuple(replicate.inner.run_dir for replicate in replicates),
        best_inner_gepa_score=best_inner.inner.gepa_score,
        best_inner_candidate_count=best_inner.inner.candidate_count,
        best_inner_run_id=best_inner.inner.run_dir,
        descriptor_representative_candidate=representative.inner.best_task_candidate,
        descriptor_representative_run_id=representative.inner.run_dir,
    )
    return descriptor, evaluation


def _minimum(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot aggregate an empty collection")
    return min(materialized)


def _recent_unique_records(
    records: Sequence[EvaluationRecord],
    limit: int,
) -> tuple[EvaluationRecord, ...]:
    selected: list[EvaluationRecord] = []
    seen: set[str] = set()
    for record in reversed(records):
        if not record.evaluation.fitness.eligible:
            continue
        if record.genome.genome_id in seen:
            continue
        seen.add(record.genome.genome_id)
        selected.append(record)
        if len(selected) == limit:
            break
    selected.reverse()
    return tuple(selected)


def _ancestral_genomes(
    record: EvaluationRecord,
    records_by_genome: Mapping[str, EvaluationRecord],
) -> tuple[MetaPromptGenome, ...]:
    """Return admitted genotypes in chronological order from root to parent."""

    newest_first: list[MetaPromptGenome] = []
    seen: set[str] = set()
    current: EvaluationRecord | None = record
    while current is not None and current.genome.genome_id not in seen:
        seen.add(current.genome.genome_id)
        newest_first.append(current.genome)
        parent_id = current.parent_genome_id
        current = None if parent_id is None else records_by_genome.get(parent_id)
    newest_first.reverse()
    return tuple(newest_first)


def _merge_unique_records(
    *groups: Sequence[EvaluationRecord],
) -> tuple[EvaluationRecord, ...]:
    merged: list[EvaluationRecord] = []
    seen: set[str] = set()
    for group in groups:
        for record in group:
            if not record.evaluation.fitness.eligible or record.genome.genome_id in seen:
                continue
            seen.add(record.genome.genome_id)
            merged.append(record)
    return tuple(merged)


def _derived_seed(master_seed: int, proposal_id: int, purpose: str) -> int:
    payload = f"{master_seed}:{proposal_id}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def _failure(
    proposal_id: int,
    stage: str,
    exc: Exception,
    *,
    parent_genome_id: str | None = None,
    genome: MetaPromptGenome | None = None,
) -> EvolutionFailure:
    return EvolutionFailure(
        proposal_id=proposal_id,
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc),
        parent_genome_id=parent_genome_id,
        genome_id=None if genome is None else genome.genome_id,
    )


def _initialize_manifest(
    run_root: Path,
    archive: MapElitesArchive,
    config: EvolutionConfig,
) -> None:
    path = run_root / "manifest.json"
    if path.exists():
        raise FileExistsError(
            f"run directory already contains {path.name}; choose a fresh run directory"
        )
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "archive_individual": "gepa_reflection_meta_prompt",
            "descriptor_hasher_version": archive.hasher_version,
            "possible_keys": (
                list(archive.possible_keys) if archive.possible_keys is not None else None
            ),
            "offspring_count": config.offspring_count,
            "master_seed": config.master_seed,
            "replicate_seeds": list(config.replicate_seeds),
            "lamarckian_success_threshold": config.lamarckian_success_threshold,
            "problem_description": config.problem_description,
        },
    )


def _persist_state(
    run_root: Path,
    archive: MapElitesArchive,
    records: Sequence[EvaluationRecord],
    failures: Sequence[EvolutionFailure],
    *,
    metric_calls_consumed: int,
) -> None:
    history = {
        "schema_version": 1,
        "records": [_record_dict(record) for record in records],
        "failures": [
            {
                "proposal_id": failure.proposal_id,
                "stage": failure.stage,
                "error_type": failure.error_type,
                "message": failure.message,
                "parent_genome_id": failure.parent_genome_id,
                "genome_id": failure.genome_id,
            }
            for failure in failures
        ],
        "metric_calls_consumed": metric_calls_consumed,
    }
    archive_snapshot = archive.snapshot()
    # checkpoint.json is the authoritative, single-file transaction. The two
    # projection files that follow are convenient for existing archive tools
    # and human inspection; a crash between them cannot invalidate checkpoint.
    _atomic_json(
        run_root / "checkpoint.json",
        {
            "schema_version": 1,
            "archive": archive_snapshot,
            "history": history,
        },
    )
    archive.save(run_root / "archive.json")
    _atomic_json(run_root / "history.json", history)


def _record_dict(record: EvaluationRecord) -> dict[str, Any]:
    mutation_trace = (
        None
        if record.mutation is None
        else {
            "operator": record.mutation.operator.value,
            "lineage": list(record.mutation.lineage),
            "raw_prompts": list(record.mutation.raw_prompts),
            "raw_outputs": list(record.mutation.raw_outputs),
            "call_seeds": list(record.mutation.call_seeds),
            "repaired_placeholders": list(record.mutation.repaired_placeholders),
        }
    )
    return {
        "proposal_id": record.proposal_id,
        "genome": record.genome.to_dict(),
        "descriptor": record.descriptor.to_dict(),
        "bin_key": record.admission.key,
        "quality": record.evaluation.quality,
        "evaluation": record.evaluation.to_dict(),
        "accepted": record.admission.accepted,
        "admission_reason": record.admission.reason.value,
        "parent_genome_id": record.parent_genome_id,
        "operator": None if record.mutation is None else record.mutation.operator.value,
        "lineage": list(record.lineage),
        "mutation_trace": mutation_trace,
        "from_cache": record.from_cache,
        "replicates": [
            {
                "seed": replicate.seed,
                "inner": replicate.inner.to_dict(),
                "descriptor": replicate.downstream.descriptor.to_dict(),
                "fitness": replicate.downstream.fitness.to_dict(),
            }
            for replicate in record.replicates
        ],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_path)
        raise


__all__ = [
    "EvaluationRecord",
    "EvolutionConfig",
    "EvolutionFailure",
    "EvolutionPersistenceError",
    "EvolutionResult",
    "GenomeDigestCollisionError",
    "InnerRunEvidence",
    "InnerRunner",
    "ReplicateEvaluation",
    "evolve_meta_prompts",
]
