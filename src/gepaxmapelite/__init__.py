"""Public API for nested GEPA, Promptbreeder, and MAP-Elites research."""

from gepaxmapelite.archive import (
    AdmissionDecision,
    AdmissionReason,
    ArchiveError,
    HasherVersionMismatchError,
    MapElitesArchive,
    MetaPromptElite,
)
from gepaxmapelite.engine import (
    EvaluationRecord,
    EvolutionConfig,
    EvolutionFailure,
    EvolutionPersistenceError,
    EvolutionResult,
    GenomeDigestCollisionError,
    InnerRunEvidence,
    ReplicateEvaluation,
    evolve_meta_prompts,
)
from gepaxmapelite.evaluation import (
    AmbiguityNoiseEvaluator,
    AmbiguityNoiseProbe,
    DownstreamEvaluation,
    ProbeEvidence,
    ProbeJudgment,
    ProbeStratum,
)
from gepaxmapelite.gepa_runner import (
    GEPAInnerRunFailure,
    GEPAInnerRunner,
    InnerRunResult,
    run_gepa_inner,
)
from gepaxmapelite.hashing import (
    AmbiguityNoiseDescriptorHasher,
    CallableDescriptorHasher,
    DefaultBehaviorDescriptorHasher,
    DescriptorHasher,
    DescriptorKey,
    default_descriptor_key,
    descriptor_bin_index,
)
from gepaxmapelite.models import (
    AmbiguityStrategy,
    BehaviorDescriptor,
    FitnessReport,
    FrozenPromptCandidate,
    MetaPromptEvaluation,
    MetaPromptGenome,
    NoiseStrategy,
)
from gepaxmapelite.promptbreeder import (
    MutationOperator,
    MutationResult,
    PromptBreeder,
)
from gepaxmapelite.text_adapter import (
    TextCleanupAdapter,
    TextExample,
    TextJudgment,
    TextTrajectory,
    adapter_factory,
)

__version__ = "0.1.0"

__all__ = [
    "AdmissionDecision",
    "AdmissionReason",
    "AmbiguityNoiseDescriptorHasher",
    "AmbiguityNoiseEvaluator",
    "AmbiguityNoiseProbe",
    "AmbiguityStrategy",
    "ArchiveError",
    "BehaviorDescriptor",
    "CallableDescriptorHasher",
    "DefaultBehaviorDescriptorHasher",
    "DescriptorHasher",
    "DescriptorKey",
    "DownstreamEvaluation",
    "EvaluationRecord",
    "EvolutionConfig",
    "EvolutionFailure",
    "EvolutionPersistenceError",
    "EvolutionResult",
    "FitnessReport",
    "FrozenPromptCandidate",
    "GEPAInnerRunFailure",
    "GEPAInnerRunner",
    "GenomeDigestCollisionError",
    "HasherVersionMismatchError",
    "InnerRunEvidence",
    "InnerRunResult",
    "MapElitesArchive",
    "MetaPromptElite",
    "MetaPromptEvaluation",
    "MetaPromptGenome",
    "MutationOperator",
    "MutationResult",
    "NoiseStrategy",
    "ProbeEvidence",
    "ProbeJudgment",
    "ProbeStratum",
    "PromptBreeder",
    "ReplicateEvaluation",
    "TextCleanupAdapter",
    "TextExample",
    "TextJudgment",
    "TextTrajectory",
    "adapter_factory",
    "default_descriptor_key",
    "descriptor_bin_index",
    "evolve_meta_prompts",
    "run_gepa_inner",
]
