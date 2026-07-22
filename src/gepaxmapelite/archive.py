"""A dict-backed, versioned MAP-Elites archive for outer meta-prompts."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, cast

from gepaxmapelite.hashing import (
    DefaultBehaviorDescriptorHasher,
    DescriptorHasher,
    DescriptorKey,
)
from gepaxmapelite.models import (
    BehaviorDescriptor,
    MetaPromptEvaluation,
    MetaPromptGenome,
)


class ArchiveError(ValueError):
    """Base exception for invalid archive operations or persisted state."""


class HasherVersionMismatchError(ArchiveError):
    """Raised when persisted cells were built with another descriptor hasher."""


class AdmissionReason(str, Enum):
    FILLED_EMPTY_CELL = "filled_empty_cell"
    REPLACED_INCUMBENT = "replaced_incumbent"
    NOT_BETTER = "not_better"
    GUARDRAIL_VIOLATION = "guardrail_violation"


@dataclass(frozen=True, slots=True)
class MetaPromptElite:
    """An atomic archive value: genome and its own immutable evaluation."""

    key: DescriptorKey
    descriptor: BehaviorDescriptor
    genome: MetaPromptGenome
    evaluation: MetaPromptEvaluation

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("elite key must be a non-empty string")
        if not isinstance(self.descriptor, BehaviorDescriptor):
            raise TypeError("elite descriptor must be a BehaviorDescriptor")
        if not isinstance(self.genome, MetaPromptGenome):
            raise TypeError("elite genome must be a MetaPromptGenome")
        if not isinstance(self.evaluation, MetaPromptEvaluation):
            raise TypeError("elite evaluation must be a MetaPromptEvaluation")
        if self.evaluation.genome_id != self.genome.genome_id:
            raise ValueError("elite evaluation belongs to a different meta-prompt genome")

    @property
    def quality(self) -> float:
        return self.evaluation.quality

    @property
    def meta_prompt(self) -> str:
        """The archived GEPA reflection prompt that generated task prompts."""

        return self.genome.reflection_template

    @property
    def best_task_candidate(self) -> Mapping[str, str]:
        """Highest-scoring inner GEPA result; this is not the archived individual."""

        return self.evaluation.best_inner_candidate

    @property
    def descriptor_task_candidate(self) -> Mapping[str, str]:
        """Task candidate selected from the replicate group defining this cell."""

        candidate = self.evaluation.descriptor_representative_candidate
        assert candidate is not None
        return candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "descriptor": self.descriptor.to_dict(),
            "genome": self.genome.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MetaPromptElite:
        return cls(
            key=value["key"],
            descriptor=BehaviorDescriptor.from_dict(value["descriptor"]),
            genome=MetaPromptGenome.from_dict(value["genome"]),
            evaluation=MetaPromptEvaluation.from_dict(value["evaluation"]),
        )


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    reason: AdmissionReason
    key: DescriptorKey
    elite: MetaPromptElite | None
    displaced: MetaPromptElite | None = None


def _validate_hasher(hasher: DescriptorHasher) -> tuple[str, tuple[str, ...] | None]:
    version = getattr(hasher, "version", None)
    if not isinstance(version, str) or not version.strip():
        raise TypeError("descriptor hasher must expose a non-empty string version")

    possible = getattr(hasher, "possible_keys", None)
    if possible is None:
        return version, None
    if isinstance(possible, str):
        raise TypeError("possible_keys must be a collection of keys, not one string")
    keys = tuple(possible)
    if not keys:
        raise ValueError("possible_keys must be non-empty when supplied")
    if any(not isinstance(key, str) or not key for key in keys):
        raise TypeError("all possible descriptor keys must be non-empty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("possible_keys contains duplicates")
    return version, tuple(sorted(keys))


class MapElitesArchive:
    """Thread-safe, one-elite-per-key archive for meta-prompt genomes."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        hasher: DescriptorHasher | None = None,
        *,
        replacement_epsilon: float = 1e-12,
    ) -> None:
        active_hasher: DescriptorHasher = (
            hasher
            if hasher is not None
            else cast(DescriptorHasher, DefaultBehaviorDescriptorHasher())
        )
        version, possible_keys = _validate_hasher(active_hasher)
        if isinstance(replacement_epsilon, bool) or not isinstance(
            replacement_epsilon, (int, float)
        ):
            raise TypeError("replacement_epsilon must be a real number")
        epsilon = float(replacement_epsilon)
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("replacement_epsilon must be finite and non-negative")

        self._hasher = active_hasher
        self._hasher_version = version
        self._possible_keys = possible_keys
        self._replacement_epsilon = epsilon
        self._elites: dict[DescriptorKey, MetaPromptElite] = {}
        self._lock = RLock()

    @property
    def hasher_version(self) -> str:
        return self._hasher_version

    @property
    def possible_keys(self) -> tuple[str, ...] | None:
        return self._possible_keys

    @property
    def replacement_epsilon(self) -> float:
        return self._replacement_epsilon

    @property
    def occupied_count(self) -> int:
        with self._lock:
            return len(self._elites)

    @property
    def coverage(self) -> float | None:
        if self._possible_keys is None:
            return None
        with self._lock:
            return len(self._elites) / len(self._possible_keys)

    @property
    def qd_score(self) -> float:
        with self._lock:
            return sum(elite.quality for elite in self._elites.values())

    def __len__(self) -> int:
        return self.occupied_count

    def __iter__(self) -> Iterator[DescriptorKey]:
        with self._lock:
            keys = tuple(sorted(self._elites))
        return iter(keys)

    def _key_for(self, descriptor: BehaviorDescriptor) -> DescriptorKey:
        key = self._hasher.key(descriptor)
        if not isinstance(key, str) or not key:
            raise ArchiveError("descriptor hasher returned a non-string or empty key")
        if self._possible_keys is not None and key not in self._possible_keys:
            raise ArchiveError(
                f"descriptor hasher returned key {key!r}, which is not in possible_keys"
            )
        return key

    def elite(self, key: DescriptorKey) -> MetaPromptElite | None:
        with self._lock:
            return self._elites.get(key)

    def elite_for(self, descriptor: BehaviorDescriptor) -> MetaPromptElite | None:
        return self.elite(self._key_for(descriptor))

    def elites(self) -> dict[DescriptorKey, MetaPromptElite]:
        """Return a shallow copy; elite values are immutable."""

        with self._lock:
            return dict(self._elites)

    def _is_better(self, candidate: MetaPromptElite, incumbent: MetaPromptElite) -> bool:
        quality_delta = candidate.quality - incumbent.quality
        if quality_delta > self._replacement_epsilon:
            return True
        if quality_delta < -self._replacement_epsilon:
            return False

        cost_delta = candidate.evaluation.total_cost - incumbent.evaluation.total_cost
        if cost_delta < -self._replacement_epsilon:
            return True
        if cost_delta > self._replacement_epsilon:
            return False

        variance_delta = (
            candidate.evaluation.quality_variance - incumbent.evaluation.quality_variance
        )
        if variance_delta < -self._replacement_epsilon:
            return True
        if variance_delta > self._replacement_epsilon:
            return False

        return candidate.genome.genome_id < incumbent.genome.genome_id

    def insert(
        self,
        genome: MetaPromptGenome,
        evaluation: MetaPromptEvaluation,
        descriptor: BehaviorDescriptor,
    ) -> AdmissionDecision:
        """Atomically consider a complete genome/evaluation pair for one cell."""

        if not isinstance(genome, MetaPromptGenome):
            raise TypeError("genome must be a MetaPromptGenome")
        if not isinstance(evaluation, MetaPromptEvaluation):
            raise TypeError("evaluation must be a MetaPromptEvaluation")
        if evaluation.genome_id != genome.genome_id:
            raise ValueError("evaluation belongs to a different meta-prompt genome")
        if not isinstance(descriptor, BehaviorDescriptor):
            raise TypeError("descriptor must be a BehaviorDescriptor")

        key = self._key_for(descriptor)
        candidate = MetaPromptElite(
            key=key,
            descriptor=descriptor,
            genome=genome,
            evaluation=evaluation,
        )

        with self._lock:
            incumbent = self._elites.get(key)
            if not evaluation.fitness.eligible:
                return AdmissionDecision(
                    accepted=False,
                    reason=AdmissionReason.GUARDRAIL_VIOLATION,
                    key=key,
                    elite=incumbent,
                )
            if incumbent is None:
                self._elites[key] = candidate
                return AdmissionDecision(
                    accepted=True,
                    reason=AdmissionReason.FILLED_EMPTY_CELL,
                    key=key,
                    elite=candidate,
                )
            if self._is_better(candidate, incumbent):
                self._elites[key] = candidate
                return AdmissionDecision(
                    accepted=True,
                    reason=AdmissionReason.REPLACED_INCUMBENT,
                    key=key,
                    elite=candidate,
                    displaced=incumbent,
                )
            return AdmissionDecision(
                accepted=False,
                reason=AdmissionReason.NOT_BETTER,
                key=key,
                elite=incumbent,
            )

    def consider(
        self,
        genome: MetaPromptGenome,
        evaluation: MetaPromptEvaluation,
        descriptor: BehaviorDescriptor,
    ) -> AdmissionDecision:
        return self.insert(genome, evaluation, descriptor)

    def select_uniform(self, rng: random.Random | None = None) -> MetaPromptElite:
        """Select uniformly across occupied descriptor keys."""

        active_rng = rng or random.Random()
        with self._lock:
            keys = tuple(sorted(self._elites))
            if not keys:
                raise LookupError("cannot select from an empty MAP-Elites archive")
            key = keys[active_rng.randrange(len(keys))]
            return self._elites[key]

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe, internally consistent archive snapshot."""

        with self._lock:
            elites = {key: self._elites[key].to_dict() for key in sorted(self._elites)}
            return {
                "schema_version": self.SCHEMA_VERSION,
                "hasher_version": self._hasher_version,
                "possible_keys": (
                    list(self._possible_keys) if self._possible_keys is not None else None
                ),
                "replacement_epsilon": self._replacement_epsilon,
                "elites": elites,
            }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        hasher: DescriptorHasher | None = None,
    ) -> MapElitesArchive:
        if not isinstance(snapshot, Mapping):
            raise TypeError("archive snapshot must be a mapping")
        if snapshot.get("schema_version") != cls.SCHEMA_VERSION:
            raise ArchiveError(
                f"unsupported archive schema version: {snapshot.get('schema_version')!r}"
            )

        stored_version = snapshot.get("hasher_version")
        if not isinstance(stored_version, str) or not stored_version:
            raise ArchiveError("archive snapshot is missing hasher_version")

        active_hasher: DescriptorHasher = (
            hasher
            if hasher is not None
            else cast(DescriptorHasher, DefaultBehaviorDescriptorHasher())
        )
        active_version, active_possible = _validate_hasher(active_hasher)
        if active_version != stored_version:
            raise HasherVersionMismatchError(
                f"archive uses descriptor hasher {stored_version!r}, "
                f"but {active_version!r} was supplied"
            )

        stored_possible_value = snapshot.get("possible_keys")
        stored_possible = None if stored_possible_value is None else tuple(stored_possible_value)
        if stored_possible != active_possible:
            raise HasherVersionMismatchError(
                "archive possible_keys do not match the supplied descriptor hasher"
            )

        archive = cls(
            hasher=active_hasher,
            replacement_epsilon=snapshot.get("replacement_epsilon", 1e-12),
        )
        persisted_elites = snapshot.get("elites")
        if not isinstance(persisted_elites, Mapping):
            raise ArchiveError("archive snapshot elites must be a mapping")

        restored: dict[str, MetaPromptElite] = {}
        for persisted_key, value in persisted_elites.items():
            if not isinstance(persisted_key, str):
                raise ArchiveError("persisted archive keys must be strings")
            if not isinstance(value, Mapping):
                raise ArchiveError(f"elite {persisted_key!r} must be a mapping")
            elite = MetaPromptElite.from_dict(value)
            computed_key = archive._key_for(elite.descriptor)
            if persisted_key != elite.key or persisted_key != computed_key:
                raise ArchiveError(f"elite key {persisted_key!r} does not match its descriptor")
            if not elite.evaluation.fitness.eligible:
                raise ArchiveError(f"persisted elite {persisted_key!r} has guardrail violations")
            restored[persisted_key] = elite

        with archive._lock:
            archive._elites = restored
        return archive

    load_snapshot = from_snapshot

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save a canonical JSON snapshot to ``path``."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.snapshot(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)
            raise

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        hasher: DescriptorHasher | None = None,
    ) -> MapElitesArchive:
        with Path(path).open("r", encoding="utf-8") as stream:
            snapshot = json.load(stream)
        return cls.from_snapshot(snapshot, hasher=hasher)


__all__ = [
    "AdmissionDecision",
    "AdmissionReason",
    "ArchiveError",
    "HasherVersionMismatchError",
    "MapElitesArchive",
    "MetaPromptElite",
]
