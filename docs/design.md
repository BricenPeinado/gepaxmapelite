# GEPA × Promptbreeder × MAP-Elites

This project uses a nested search. Promptbreeder evolves the **reflection
meta-prompt** that GEPA uses when it proposes a new downstream task prompt.
Each meta-prompt is evaluated in a fresh inner GEPA optimization. The best task
prompt from that run is tested on held-out ambiguity/noise probes, and the
resulting behavior determines which MAP-Elites cell the **meta-prompt** competes
for.

```text
Promptbreeder meta-prompt
    -> isolated GEPA run
    -> best generated task prompt
    -> held-out behavior and fitness
    -> custom behavior hash
    -> MAP-Elites cell containing the meta-prompt
```

The downstream task prompt is evidence attached to an elite. It is never the
archived genome and never becomes a Promptbreeder parent.

## Why 16 bins

The default map crosses two observable policy dimensions with four levels each.
This gives 16 niches: enough to preserve meaningfully different cleanup
strategies, while still being fillable under the cost of a nested LLM search.

### Ambiguity policy

| Level | Label | Observable behavior |
| --- | --- | --- |
| A0 | Resolve | Choose the most likely reading and answer directly. |
| A1 | State assumptions | Choose a reading but explicitly expose the assumptions. |
| A2 | Preserve branches | Retain or answer multiple plausible interpretations. |
| A3 | Clarify | Ask for missing information or abstain when guessing is unsafe. |

### Noise policy

| Level | Label | Observable behavior |
| --- | --- | --- |
| N0 | Surface cleanup | Fix formatting, spelling, and obvious local corruption. |
| N1 | Relevance filter | Remove distractors while retaining the original structure. |
| N2 | Extract/normalize | Recover facts and constraints into a clearer structure. |
| N3 | Verify/reconcile | Cross-check conflicts, mark uncertainty, and avoid invented repairs. |

The default custom hash is collision-free over these descriptors:

```text
bin = 4 * ambiguity_level + noise_level
```

It returns decimal-string keys `"0".."15"` and does not call Python's randomized
`hash()`. Multiple meta-prompts intentionally map to the same key and compete
there. A different stable descriptor hasher can be injected for another
research hypothesis.

## Fitness and behavior are separate

The descriptor answers *what policy does the generated task prompt exhibit?*
Fitness answers *how well does it work?* Keeping them separate prevents the
map from collapsing to a single generic prompt.

The default recommendation is the worst held-out stratum score:

```text
quality = min(clean, ambiguity, noise, mixed, meaning_preservation)
```

The injected held-out judge should report any changed or invented fact,
constraint, entity, or intent as a guardrail violation. Reported violations
make an individual ineligible for the archive, preventing a prompt from scoring
well merely by deleting difficult content.

With multiple replicate seeds, each stratum uses its worst replicate score.
The cell descriptor is the modal joint behavior; a frequency tie is resolved by
the best-performing descriptor group. The evaluation stores two explicit task
candidates: `best_inner_candidate` is the globally highest-scoring GEPA result,
while `descriptor_representative_candidate` is selected from the descriptor
group defining the cell. The latter's observed behavior therefore agrees with
the cell without mislabeling it as the globally best inner run.

## Promptbreeder adaptation

The original Promptbreeder search evolves task prompts and the mutation prompts
that mutate them. Here, its task-level individual is deliberately moved up one
level: it is GEPA's reflection template. A genome contains:

- `reflection_template`: the archived prompt that GEPA uses to generate task
  prompts; it must retain `<curr_param>` and `<side_info>`.
- `mutation_prompt`: the self-referential instruction Promptbreeder uses to
  mutate that reflection template.

Direct mutation applies the mutation prompt to the reflection template.
First-order hypermutation first improves the mutation prompt and then uses the
new mutation prompt to produce a new reflection template. Population, lineage,
ranked EDA, zero-order hypermutation, and Lamarckian operators provide broader
exploration while preserving the same archive boundary.

Automatic operator selection renormalizes weights over operators whose required
context is present, so Lamarckian mutation cannot consume an offspring slot
when no successful held-out evidence exists. EDA reference lists are
deduplicated so an active elite that also appears in the recent population is
not double-weighted.

## Isolation invariant

Every meta-prompt evaluation starts from the same downstream seed prompt,
datasets, budget, and replicate seeds in a new GEPA run directory with fresh
adapter and model state. GEPA resumes from an existing run directory, so sharing
directories across meta-prompts would leak candidate history and invalidate the
comparison.

The inner wrapper also requires at least one successful GEPA proposal callback.
This matters because GEPA 0.1.4 can catch a reflection-provider exception and
return a normal-looking seed-only result. Such a result is rejected here rather
than treated as evidence about the meta-prompt.
