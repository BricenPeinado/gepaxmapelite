"""Held-out ambiguity/noise behavior evaluation for generated task prompts."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from gepaxmapelite.models import (
    AmbiguityStrategy,
    BehaviorDescriptor,
    FitnessReport,
    NoiseStrategy,
)


class ProbeStratum(str, Enum):
    """The five held-out strata used by the worst-group objective."""

    CLEAN = "clean"
    AMBIGUITY = "ambiguity"
    NOISE = "noise"
    MIXED = "mixed"
    MEANING_PRESERVATION = "meaning_preservation"


@dataclass(frozen=True, slots=True)
class AmbiguityNoiseProbe:
    """One held-out input and the information a judge needs to score it."""

    probe_id: str
    text: str
    stratum: ProbeStratum
    rubric: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.probe_id, str) or not self.probe_id:
            raise ValueError("probe_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("probe text must be a non-empty string")
        if not isinstance(self.stratum, ProbeStratum):
            object.__setattr__(self, "stratum", ProbeStratum(self.stratum))
        if not isinstance(self.rubric, str):
            raise TypeError("probe rubric must be a string")


@dataclass(frozen=True, slots=True)
class ProbeJudgment:
    """A judge's score and classification of observable cleanup behavior."""

    score: float
    feedback: str = ""
    ambiguity_strategy: AmbiguityStrategy | None = None
    noise_strategy: NoiseStrategy | None = None
    guardrail_violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("probe score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("probe score must be finite and in [0, 1]")
        object.__setattr__(self, "score", score)
        if not isinstance(self.feedback, str):
            raise TypeError("probe feedback must be a string")
        if self.ambiguity_strategy is not None:
            object.__setattr__(
                self,
                "ambiguity_strategy",
                AmbiguityStrategy(self.ambiguity_strategy),
            )
        if self.noise_strategy is not None:
            object.__setattr__(self, "noise_strategy", NoiseStrategy(self.noise_strategy))
        violations = tuple(self.guardrail_violations)
        if any(not isinstance(item, str) or not item for item in violations):
            raise ValueError("guardrail violations must be non-empty strings")
        object.__setattr__(self, "guardrail_violations", tuple(dict.fromkeys(violations)))


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    probe_id: str
    stratum: ProbeStratum
    output: str
    judgment: ProbeJudgment


@dataclass(frozen=True, slots=True)
class DownstreamEvaluation:
    """Fitness and descriptor derived from one generated downstream prompt."""

    fitness: FitnessReport
    descriptor: BehaviorDescriptor
    evidence: tuple[ProbeEvidence, ...]

    @property
    def feedback(self) -> tuple[str, ...]:
        return tuple(item.judgment.feedback for item in self.evidence if item.judgment.feedback)


class ProbeExecutor(Protocol):
    """Execute a structured GEPA task candidate on one held-out probe."""

    def __call__(
        self,
        candidate: Mapping[str, str],
        probe: AmbiguityNoiseProbe,
        *,
        seed: int,
    ) -> str: ...


class ProbeJudge(Protocol):
    """Score and behavior-classify a held-out task-model output."""

    def __call__(
        self,
        probe: AmbiguityNoiseProbe,
        output: str,
        *,
        seed: int,
    ) -> ProbeJudgment: ...


class DownstreamEvaluator(Protocol):
    """Outer evaluator contract consumed by the evolution engine."""

    def evaluate(
        self,
        candidate: Mapping[str, str],
        *,
        replicate_seed: int,
    ) -> DownstreamEvaluation: ...


class AmbiguityNoiseEvaluator:
    """Evaluate a GEPA-generated task prompt on a fixed held-out probe suite.

    Behavior bins come from the observed outputs, not keyword inspection of the
    meta-prompt. For each axis the dominant policy on the relevant strata is
    selected; ties go to the higher (more conservative) ordinal strategy.
    """

    def __init__(
        self,
        probes: Sequence[AmbiguityNoiseProbe],
        executor: ProbeExecutor,
        judge: ProbeJudge,
    ) -> None:
        if not callable(executor) or not callable(judge):
            raise TypeError("executor and judge must be callable")
        self._probes = tuple(probes)
        if not self._probes:
            raise ValueError("the held-out probe suite cannot be empty")
        ids = [probe.probe_id for probe in self._probes]
        if len(set(ids)) != len(ids):
            raise ValueError("held-out probe ids must be unique")
        present = {probe.stratum for probe in self._probes}
        missing = set(ProbeStratum) - present
        if missing:
            labels = ", ".join(sorted(stratum.value for stratum in missing))
            raise ValueError(f"the held-out probe suite is missing strata: {labels}")
        self._executor = executor
        self._judge = judge

    def evaluate(
        self,
        candidate: Mapping[str, str],
        *,
        replicate_seed: int,
    ) -> DownstreamEvaluation:
        if isinstance(replicate_seed, bool) or not isinstance(replicate_seed, int):
            raise TypeError("replicate_seed must be an integer")
        if not candidate or any(
            not isinstance(name, str) or not isinstance(text, str)
            for name, text in candidate.items()
        ):
            raise ValueError("candidate must be a non-empty string-to-string mapping")

        evidence: list[ProbeEvidence] = []
        by_stratum: dict[ProbeStratum, list[float]] = defaultdict(list)
        ambiguity_observations: list[AmbiguityStrategy] = []
        noise_observations: list[NoiseStrategy] = []
        violations: list[str] = []

        for probe in self._probes:
            probe_seed = _derive_probe_seed(replicate_seed, probe.probe_id)
            try:
                output = self._executor(candidate, probe, seed=probe_seed)
                if not isinstance(output, str):
                    raise TypeError("probe executor must return str")
                judgment = self._judge(probe, output, seed=probe_seed)
                if not isinstance(judgment, ProbeJudgment):
                    raise TypeError("probe judge must return ProbeJudgment")
            except Exception as exc:
                output = ""
                judgment = ProbeJudgment(
                    score=0.0,
                    feedback=f"Probe failed: {type(exc).__name__}: {exc}",
                    guardrail_violations=(f"probe_error:{probe.probe_id}",),
                )

            if probe.stratum in (ProbeStratum.AMBIGUITY, ProbeStratum.MIXED):
                if judgment.ambiguity_strategy is None:
                    violations.append(f"missing_ambiguity_descriptor:{probe.probe_id}")
                else:
                    ambiguity_observations.append(judgment.ambiguity_strategy)
            if probe.stratum in (ProbeStratum.NOISE, ProbeStratum.MIXED):
                if judgment.noise_strategy is None:
                    violations.append(f"missing_noise_descriptor:{probe.probe_id}")
                else:
                    noise_observations.append(judgment.noise_strategy)

            by_stratum[probe.stratum].append(judgment.score)
            violations.extend(
                f"{probe.probe_id}:{violation}" for violation in judgment.guardrail_violations
            )
            evidence.append(ProbeEvidence(probe.probe_id, probe.stratum, output, judgment))

        # Missing observations are already hard failures; a deterministic default
        # keeps the failed evaluation serializable while preventing archive entry.
        ambiguity = _dominant_strategy(ambiguity_observations, AmbiguityStrategy.RESOLVE)
        noise = _dominant_strategy(noise_observations, NoiseStrategy.SURFACE_CLEANUP)
        fitness = FitnessReport(
            clean_score=_mean(by_stratum[ProbeStratum.CLEAN]),
            ambiguity_score=_mean(by_stratum[ProbeStratum.AMBIGUITY]),
            noise_score=_mean(by_stratum[ProbeStratum.NOISE]),
            mixed_score=_mean(by_stratum[ProbeStratum.MIXED]),
            meaning_preservation_score=_mean(by_stratum[ProbeStratum.MEANING_PRESERVATION]),
            guardrail_violations=tuple(dict.fromkeys(violations)),
        )
        return DownstreamEvaluation(
            fitness=fitness,
            descriptor=BehaviorDescriptor(ambiguity=ambiguity, noise=noise),
            evidence=tuple(evidence),
        )


def _derive_probe_seed(replicate_seed: int, probe_id: str) -> int:
    payload = f"{replicate_seed}:probe:{probe_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def _dominant_strategy(values: Sequence[Any], default: Any) -> Any:
    if not values:
        return default
    counts = Counter(values)
    largest = max(counts.values())
    return max(value for value, count in counts.items() if count == largest)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty probe stratum")
    return sum(values) / len(values)


__all__ = [
    "AmbiguityNoiseEvaluator",
    "AmbiguityNoiseProbe",
    "DownstreamEvaluation",
    "DownstreamEvaluator",
    "ProbeEvidence",
    "ProbeExecutor",
    "ProbeJudge",
    "ProbeJudgment",
    "ProbeStratum",
]
