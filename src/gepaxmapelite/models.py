"""Immutable domain models for nested GEPA x MAP-Elites optimization.

The outer evolutionary artifact is a :class:`MetaPromptGenome`.  Evaluating
that genome runs an isolated inner GEPA optimization and produces a
:class:`MetaPromptEvaluation` containing the best structured downstream
candidate and its held-out fitness.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

_GEPA_REFLECTION_PLACEHOLDERS = ("<curr_param>", "<side_info>")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    """Return the canonical JSON representation used for stable identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalise_score(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return result


def _normalise_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


class AmbiguityStrategy(IntEnum):
    """Dominant strategy used to handle ambiguous source text."""

    RESOLVE = 0
    ASSUME = 1
    BRANCH = 2
    CLARIFY = 3

    # Descriptive aliases retain the compact public names above while making
    # research logs self-explanatory.
    DIRECT_RESOLUTION = RESOLVE
    STATE_ASSUMPTIONS = ASSUME
    PRESERVE_BRANCHES = BRANCH
    CLARIFY_OR_ABSTAIN = CLARIFY


class NoiseStrategy(IntEnum):
    """Dominant strategy used to handle noisy source text."""

    SURFACE_CLEANUP = 0
    FILTER = 1
    EXTRACT_NORMALIZE = 2
    VERIFY_REVISE = 3

    EXTRACT = EXTRACT_NORMALIZE
    VERIFY = VERIFY_REVISE


@dataclass(frozen=True, slots=True)
class BehaviorDescriptor:
    """The two-dimensional behavioral descriptor used by the default map."""

    ambiguity: AmbiguityStrategy
    noise: NoiseStrategy

    def __post_init__(self) -> None:
        if isinstance(self.ambiguity, bool):
            raise TypeError("ambiguity must be an AmbiguityStrategy, not bool")
        if isinstance(self.noise, bool):
            raise TypeError("noise must be a NoiseStrategy, not bool")
        try:
            ambiguity = AmbiguityStrategy(self.ambiguity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid ambiguity strategy: {self.ambiguity!r}") from exc
        try:
            noise = NoiseStrategy(self.noise)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid noise strategy: {self.noise!r}") from exc
        object.__setattr__(self, "ambiguity", ambiguity)
        object.__setattr__(self, "noise", noise)

    def to_dict(self) -> dict[str, int]:
        return {"ambiguity": int(self.ambiguity), "noise": int(self.noise)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BehaviorDescriptor:
        return cls(
            ambiguity=AmbiguityStrategy(value["ambiguity"]),
            noise=NoiseStrategy(value["noise"]),
        )


@dataclass(frozen=True, slots=True)
class MetaPromptGenome:
    """A PromptBreeder unit whose reflection template configures inner GEPA.

    ``genome_id`` is a stable SHA-256 digest over canonical JSON containing
    both prompts.  Prompt whitespace is intentionally significant.
    """

    reflection_template: str
    mutation_prompt: str
    genome_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reflection_template, str):
            raise TypeError("reflection_template must be a string")
        if not self.reflection_template.strip():
            raise ValueError("reflection_template must not be empty")
        if not isinstance(self.mutation_prompt, str):
            raise TypeError("mutation_prompt must be a string")
        if not self.mutation_prompt.strip():
            raise ValueError("mutation_prompt must not be empty")

        missing = [
            placeholder
            for placeholder in _GEPA_REFLECTION_PLACEHOLDERS
            if placeholder not in self.reflection_template
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"reflection_template is missing required GEPA placeholder(s): {joined}"
            )

        payload = {
            "mutation_prompt": self.mutation_prompt,
            "reflection_template": self.reflection_template,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        object.__setattr__(self, "genome_id", digest)

    @property
    def identity(self) -> str:
        """Alias for callers that treat the digest as a generic identity."""

        return self.genome_id

    def to_dict(self) -> dict[str, str]:
        return {
            "reflection_template": self.reflection_template,
            "mutation_prompt": self.mutation_prompt,
            "genome_id": self.genome_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MetaPromptGenome:
        genome = cls(
            reflection_template=value["reflection_template"],
            mutation_prompt=value["mutation_prompt"],
        )
        persisted_id = value.get("genome_id")
        if persisted_id is not None and persisted_id != genome.genome_id:
            raise ValueError(
                "persisted genome_id does not match the canonical meta-prompt identity"
            )
        return genome


@dataclass(frozen=True, slots=True, init=False)
class FrozenPromptCandidate(Mapping[str, str]):
    """Hashable, immutable mapping for a structured inner GEPA candidate."""

    _items: tuple[tuple[str, str], ...]

    def __init__(
        self,
        components: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> None:
        if isinstance(components, FrozenPromptCandidate):
            items = components._items
        elif isinstance(components, Mapping):
            items = tuple(components.items())
        else:
            try:
                items = tuple(components)
            except TypeError as exc:
                raise TypeError("components must be a string-to-string mapping") from exc

        if not items:
            raise ValueError("best inner task candidate must contain at least one component")

        normalised: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("candidate components must be (name, text) pairs")
            name, text = item
            if not isinstance(name, str) or not name:
                raise TypeError("candidate component names must be non-empty strings")
            if not isinstance(text, str):
                raise TypeError(f"candidate component {name!r} must contain string text")
            if name in seen:
                raise ValueError(f"duplicate candidate component: {name!r}")
            seen.add(name)
            normalised.append((name, text))

        object.__setattr__(self, "_items", tuple(sorted(normalised)))

    def __getitem__(self, key: str) -> str:
        for name, text in self._items:
            if name == key:
                return text
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, str]:
        return dict(self._items)


@dataclass(frozen=True, slots=True)
class FitnessReport:
    """Held-out downstream performance for one outer meta-prompt.

    Scores must be normalized to ``[0, 1]``.  Quality is deliberately the
    worst of the five strata, preventing easy cases from masking a failure on
    ambiguity, noise, mixed inputs, clean controls, or meaning preservation.
    """

    clean_score: float
    ambiguity_score: float
    noise_score: float
    mixed_score: float
    meaning_preservation_score: float
    guardrail_violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "clean_score",
            "ambiguity_score",
            "noise_score",
            "mixed_score",
            "meaning_preservation_score",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_score(field_name, getattr(self, field_name)),
            )

        if isinstance(self.guardrail_violations, str):
            violations = (self.guardrail_violations,)
        else:
            violations = tuple(self.guardrail_violations)
        for violation in violations:
            if not isinstance(violation, str) or not violation.strip():
                raise ValueError("guardrail violations must be non-empty strings")
        object.__setattr__(self, "guardrail_violations", tuple(dict.fromkeys(violations)))

    @property
    def worst_stratum_quality(self) -> float:
        return min(
            self.clean_score,
            self.ambiguity_score,
            self.noise_score,
            self.mixed_score,
            self.meaning_preservation_score,
        )

    @property
    def quality(self) -> float:
        return self.worst_stratum_quality

    @property
    def eligible(self) -> bool:
        return not self.guardrail_violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_score": self.clean_score,
            "ambiguity_score": self.ambiguity_score,
            "noise_score": self.noise_score,
            "mixed_score": self.mixed_score,
            "meaning_preservation_score": self.meaning_preservation_score,
            "guardrail_violations": list(self.guardrail_violations),
            "worst_stratum_quality": self.worst_stratum_quality,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FitnessReport:
        report = cls(
            clean_score=value["clean_score"],
            ambiguity_score=value["ambiguity_score"],
            noise_score=value["noise_score"],
            mixed_score=value["mixed_score"],
            meaning_preservation_score=value["meaning_preservation_score"],
            guardrail_violations=tuple(value.get("guardrail_violations", ())),
        )
        persisted_quality = value.get("worst_stratum_quality")
        if persisted_quality is not None and not math.isclose(
            float(persisted_quality), report.worst_stratum_quality, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("persisted worst_stratum_quality does not match the five scores")
        return report


@dataclass(frozen=True, slots=True)
class MetaPromptEvaluation:
    """Result of evaluating one outer genome through isolated inner GEPA."""

    genome_id: str
    best_inner_candidate: FrozenPromptCandidate | Mapping[str, str]
    fitness: FitnessReport
    total_cost: float = 0.0
    quality_variance: float = 0.0
    inner_run_ids: tuple[str, ...] = ()
    best_inner_gepa_score: float | None = None
    best_inner_candidate_count: int | None = None
    best_inner_run_id: str | None = None
    descriptor_representative_candidate: FrozenPromptCandidate | Mapping[str, str] | None = None
    descriptor_representative_run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.genome_id, str) or not _SHA256_PATTERN.fullmatch(self.genome_id):
            raise ValueError("genome_id must be a lowercase SHA-256 hex digest")
        if not isinstance(self.best_inner_candidate, FrozenPromptCandidate):
            try:
                candidate = FrozenPromptCandidate(self.best_inner_candidate)
            except (TypeError, ValueError) as exc:
                raise TypeError("best_inner_candidate must be a structured string mapping") from exc
            object.__setattr__(self, "best_inner_candidate", candidate)
        representative = self.descriptor_representative_candidate
        representative_was_defaulted = representative is None
        if representative is None:
            representative = self.best_inner_candidate
        elif not isinstance(representative, FrozenPromptCandidate):
            try:
                representative = FrozenPromptCandidate(representative)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "descriptor_representative_candidate must be a structured string mapping"
                ) from exc
        object.__setattr__(self, "descriptor_representative_candidate", representative)
        if not isinstance(self.fitness, FitnessReport):
            raise TypeError("fitness must be a FitnessReport")
        object.__setattr__(
            self,
            "total_cost",
            _normalise_nonnegative("total_cost", self.total_cost),
        )
        object.__setattr__(
            self,
            "quality_variance",
            _normalise_nonnegative("quality_variance", self.quality_variance),
        )

        if isinstance(self.inner_run_ids, str):
            raise TypeError("inner_run_ids must be a collection of run identifiers")
        run_ids = tuple(self.inner_run_ids)
        for run_id in run_ids:
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("inner_run_ids must contain non-empty strings")
        object.__setattr__(self, "inner_run_ids", run_ids)
        if self.best_inner_gepa_score is not None:
            score = float(self.best_inner_gepa_score)
            if not math.isfinite(score):
                raise ValueError("best_inner_gepa_score must be finite")
            object.__setattr__(self, "best_inner_gepa_score", score)
        if self.best_inner_candidate_count is not None:
            count = self.best_inner_candidate_count
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("best_inner_candidate_count must be a positive integer")
        if self.best_inner_run_id is not None:
            if not isinstance(self.best_inner_run_id, str) or not self.best_inner_run_id:
                raise ValueError("best_inner_run_id must be a non-empty string")
            if self.best_inner_run_id not in run_ids:
                raise ValueError("best_inner_run_id must appear in inner_run_ids")
        representative_run_id = self.descriptor_representative_run_id
        if (
            representative_was_defaulted
            and representative_run_id is not None
            and representative_run_id != self.best_inner_run_id
        ):
            raise ValueError(
                "descriptor_representative_candidate is required when its run "
                "differs from best_inner_run_id"
            )
        if (
            representative_run_id is None
            and representative_was_defaulted
            and self.best_inner_run_id is not None
        ):
            representative_run_id = self.best_inner_run_id
            object.__setattr__(
                self,
                "descriptor_representative_run_id",
                representative_run_id,
            )
        if representative_run_id is not None:
            if not isinstance(representative_run_id, str) or not representative_run_id:
                raise ValueError("descriptor_representative_run_id must be a non-empty string")
            if representative_run_id not in run_ids:
                raise ValueError("descriptor_representative_run_id must appear in inner_run_ids")

    @property
    def quality(self) -> float:
        return self.fitness.worst_stratum_quality

    def to_dict(self) -> dict[str, Any]:
        candidate = self.best_inner_candidate
        assert isinstance(candidate, FrozenPromptCandidate)
        representative = self.descriptor_representative_candidate
        assert isinstance(representative, FrozenPromptCandidate)
        return {
            "genome_id": self.genome_id,
            "best_inner_candidate": candidate.to_dict(),
            "fitness": self.fitness.to_dict(),
            "total_cost": self.total_cost,
            "quality_variance": self.quality_variance,
            "inner_run_ids": list(self.inner_run_ids),
            "best_inner_gepa_score": self.best_inner_gepa_score,
            "best_inner_candidate_count": self.best_inner_candidate_count,
            "best_inner_run_id": self.best_inner_run_id,
            "descriptor_representative_candidate": representative.to_dict(),
            "descriptor_representative_run_id": self.descriptor_representative_run_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MetaPromptEvaluation:
        return cls(
            genome_id=value["genome_id"],
            best_inner_candidate=FrozenPromptCandidate(value["best_inner_candidate"]),
            fitness=FitnessReport.from_dict(value["fitness"]),
            total_cost=value.get("total_cost", 0.0),
            quality_variance=value.get("quality_variance", 0.0),
            # Preserve the raw container so __post_init__ can reject a single
            # string instead of silently splitting it into character IDs.
            inner_run_ids=value.get("inner_run_ids", ()),
            best_inner_gepa_score=value.get("best_inner_gepa_score"),
            best_inner_candidate_count=value.get("best_inner_candidate_count"),
            best_inner_run_id=value.get("best_inner_run_id"),
            descriptor_representative_candidate=value.get("descriptor_representative_candidate"),
            descriptor_representative_run_id=value.get("descriptor_representative_run_id"),
        )


__all__ = [
    "AmbiguityStrategy",
    "BehaviorDescriptor",
    "FitnessReport",
    "FrozenPromptCandidate",
    "MetaPromptEvaluation",
    "MetaPromptGenome",
    "NoiseStrategy",
]
