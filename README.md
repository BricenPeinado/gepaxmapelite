# gepaxmapelite

`gepaxmapelite` is a research implementation of a nested prompt optimizer:

1. **Promptbreeder** evolves the reflection/meta-prompt that governs GEPA's
   prompt-generation process, including self-referential mutation prompts.
2. **GEPA** starts a fresh inner run for every meta-prompt and evolves the
   downstream text-cleanup task prompt through reflection on execution feedback.
3. **MAP-Elites** stores the meta-prompts—not the generated task prompts—in a
   hash map of behavioral niches.

The default map has **16 bins**: four ambiguity policies crossed with four noise
policies. See [the design rationale](docs/design.md) for the bin definitions,
fitness rule, and archive boundary.

## Architecture

```text
archived GEPA reflection meta-prompt
          |
          | Promptbreeder mutation
          v
child GEPA reflection meta-prompt
          |
          | fresh, isolated GEPA optimization
          v
best generated downstream task prompt
          |
          | held-out ambiguity/noise probes
          v
behavior descriptor + worst-stratum fitness
          |
          | stable custom key: 4 * ambiguity + noise
          v
dict[str, MetaPromptElite]
```

Every archive value atomically pairs an outer `MetaPromptGenome` with evidence
produced by *that same genome's* inner GEPA runs. It retains both the globally
best inner-GEPA task candidate and the descriptor-group representative used to
explain the cell. Task candidates are never selected as Promptbreeder parents.

## Installation

```bash
python -m pip install -e '.[dev]'
```

Python 3.10+ and GEPA 0.1.4 are supported.

## Minimal experiment

Model providers are deliberately injected. A breeder LLM mutates meta-prompts;
GEPA receives fresh adapter and reflection-model instances; the held-out
executor and judge determine fitness and observed behavior.

Four valid starting reflection templates are included in
[`examples/seed_meta_prompts.json`](examples/seed_meta_prompts.json). The
offline [`examples/deterministic_demo.py`](examples/deterministic_demo.py)
exercises the full archive and replacement path without model credentials.

```python
import random

from gepaxmapelite import (
    EvolutionConfig,
    GEPAInnerRunner,
    MetaPromptGenome,
    PromptBreeder,
    evolve_meta_prompts,
)

seed = MetaPromptGenome(
    reflection_template="""You improve a downstream cleanup instruction.
Current instruction:
<curr_param>

Execution traces and evaluator feedback:
<side_info>

Preserve facts and intent. Return exactly one replacement instruction in a
triple-backtick block.""",
    mutation_prompt=(
        "Rewrite this reflection template to diagnose ambiguity and noise more "
        "precisely while preserving its runtime placeholders."
    ),
)

breeder = PromptBreeder(
    breeder_llm,                      # callable(prompt, seed=...) -> str
    rng=random.Random(7),
    problem_description="Remove ambiguity and noise without changing meaning.",
)

inner = GEPAInnerRunner(
    task_seed={"task_prompt": "Clean the text without changing its meaning."},
    trainset=train_examples,
    valset=selection_examples,
    adapter_factory=make_fresh_gepa_adapter,
    reflection_lm_factory=make_fresh_reflection_lm,
    max_metric_calls=200,
)

result = evolve_meta_prompts(
    seeds=[seed],
    breeder=breeder,
    inner_runner=inner,
    downstream_evaluator=heldout_ambiguity_noise_evaluator,
    config=EvolutionConfig(
        offspring_count=50,
        replicate_seeds=(11, 29),
        master_seed=7,
        run_dir="runs/ambiguity-noise-001",
    ),
)

for bin_key, elite in result.archive.elites().items():
    print(bin_key, elite.quality)
    print("archived meta-prompt:", elite.meta_prompt)
    print("best generated task prompt:", dict(elite.best_task_candidate))
    print("cell-representative task prompt:", dict(elite.descriptor_task_candidate))
```

`TextCleanupAdapter` provides a provider-neutral GEPA adapter for the common
single-task-prompt case. `AmbiguityNoiseEvaluator` provides the held-out
five-stratum evaluator once task execution and judging callables are supplied.
Systemic executor/judge exceptions propagate by default; list only genuinely
per-example exception classes in `recoverable_exceptions` if they should become
score-zero reflection evidence. Custom mutable GEPA data loaders must likewise
be supplied through `trainset_factory`/`valset_factory` to prevent state leaks.

## The 16 default bins

The descriptor is based on generated outputs on held-out probes—not keywords in
the prompt.

| Axis | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Ambiguity | resolve | state assumptions | preserve branches | clarify/abstain |
| Noise | surface cleanup | filter relevance | extract/normalize | verify/reconcile |

`DefaultBehaviorDescriptorHasher` maps `(A, N)` to decimal string key
`str(4*A + N)`, covering keys `"0"` through `"15"`. It never persists
Python's process-randomized `hash()` output. Supply any versioned
`DescriptorHasher` implementation to test another binning hypothesis, or wrap
a plain function with `CallableDescriptorHasher(function, version=...,
possible_keys=...)`.

## Fitness and guardrails

Each generated task prompt is scored on clean, ambiguity, noise, mixed, and
meaning-preservation strata. With replication, each stratum uses its worst
replicate; archive quality is the minimum of those five robust scores. Any
invented/changed fact, constraint, entity, or intent reported by the injected
judge is a hard guardrail violation and prevents archive admission.

## Reproducibility and artifacts

- All genomes use the same configured replicate seeds. Seed-aware adapter and
  reflection-model factories receive the seed; zero-argument factories remain
  supported when a client is deterministic by construction. A model-name
  string is materialized through GEPA's LM wrapper with the replicate seed
  forwarded to provider sampling (subject to provider support).
- Each genome × replicate gets a fresh GEPA directory, adapter, and reflection
  model; GEPA state is never resumed across meta-prompts.
- A replicate is valid only after GEPA emits a successful proposal callback;
  GEPA 0.1.4's swallowed reflection failures cannot become seed-only fitness.
- Genome IDs are SHA-256 digests of canonical structured prompt content.
- Archive keys are stable and versioned.
- `checkpoint.json` atomically stores the authoritative archive-and-history
  transaction; `history.json` and `archive.json` are inspection projections.
- The archive rejects snapshots created with a different hasher version or key
  space.

The code is research infrastructure, not a claim that one fixed 16-bin map is
universally optimal. The descriptor and held-out judge are explicit extension
points so the behavioral hypothesis can be changed and compared. The
Promptbreeder operators are a paper-inspired adaptation whose evolved
task-level artifact is deliberately moved up to GEPA's reflection-template
level.

## References

- Agrawal et al., [GEPA: Reflective Prompt Evolution Can Outperform
  Reinforcement Learning](https://arxiv.org/abs/2507.19457), 2025.
- Fernando et al., [Promptbreeder: Self-Referential Self-Improvement via Prompt
  Evolution](https://proceedings.mlr.press/v235/fernando24a.html), ICML 2024.
- Mouret and Clune, [Illuminating search spaces by mapping
  elites](https://arxiv.org/abs/1504.04909), 2015.
