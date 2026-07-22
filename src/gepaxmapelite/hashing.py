"""Stable, versioned behavior-descriptor keys for MAP-Elites."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from gepaxmapelite.models import AmbiguityStrategy, BehaviorDescriptor, NoiseStrategy

DescriptorKey = str


@runtime_checkable
class DescriptorHasher(Protocol):
    """Protocol for an injectable, persistently versioned descriptor keyer.

    Implementations must return a stable, JSON-safe string.  In particular,
    implementations must not derive persisted keys from Python's ``hash()``.
    """

    version: str
    possible_keys: Collection[DescriptorKey] | None

    def key(self, descriptor: BehaviorDescriptor) -> DescriptorKey:
        """Return the deterministic archive key for ``descriptor``."""


def descriptor_bin_index(descriptor: BehaviorDescriptor) -> int:
    """Pack two two-bit strategies into an integer in ``[0, 15]``.

    This is both ``4 * ambiguity + noise`` and ``(ambiguity << 2) | noise``.
    """

    if not isinstance(descriptor, BehaviorDescriptor):
        raise TypeError("descriptor must be a BehaviorDescriptor")
    return (int(descriptor.ambiguity) << 2) | int(descriptor.noise)


def default_descriptor_key(descriptor: BehaviorDescriptor) -> DescriptorKey:
    """Return the canonical decimal-string key for the default 16-bin map."""

    return str(descriptor_bin_index(descriptor))


@dataclass(frozen=True, slots=True)
class DefaultBehaviorDescriptorHasher:
    """Default 4 x 4 ambiguity/noise descriptor hasher."""

    version: ClassVar[str] = "ambiguity-noise-4x4-v1"
    possible_keys: ClassVar[tuple[DescriptorKey, ...]] = tuple(str(index) for index in range(16))

    def key(self, descriptor: BehaviorDescriptor) -> DescriptorKey:
        return default_descriptor_key(descriptor)

    def __call__(self, descriptor: BehaviorDescriptor) -> DescriptorKey:
        return self.key(descriptor)

    def decode(self, key: DescriptorKey) -> BehaviorDescriptor:
        if not isinstance(key, str) or key not in self.possible_keys:
            raise ValueError(f"invalid default descriptor key: {key!r}")
        index = int(key)
        return BehaviorDescriptor(
            ambiguity=AmbiguityStrategy(index >> 2),
            noise=NoiseStrategy(index & 0b11),
        )


# A concise domain-oriented alias for callers configuring an experiment.
AmbiguityNoiseDescriptorHasher = DefaultBehaviorDescriptorHasher


@dataclass(frozen=True, slots=True)
class CallableDescriptorHasher:
    """Turn a plain custom hashing function into a versioned archive hasher.

    ``version`` must change whenever the function's bin semantics or
    configuration changes, so persisted archives cannot be silently
    reinterpreted.
    """

    function: Callable[[BehaviorDescriptor], DescriptorKey]
    version: str
    possible_keys: Collection[DescriptorKey] | None = None

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")

    def key(self, descriptor: BehaviorDescriptor) -> DescriptorKey:
        return self.function(descriptor)

    def __call__(self, descriptor: BehaviorDescriptor) -> DescriptorKey:
        return self.key(descriptor)


__all__ = [
    "AmbiguityNoiseDescriptorHasher",
    "CallableDescriptorHasher",
    "DefaultBehaviorDescriptorHasher",
    "DescriptorHasher",
    "DescriptorKey",
    "default_descriptor_key",
    "descriptor_bin_index",
]
