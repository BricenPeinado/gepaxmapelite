from __future__ import annotations

import hashlib
import json
from collections.abc import Collection

import pytest

from gepaxmapelite.archive import (
    AdmissionReason,
    HasherVersionMismatchError,
    MapElitesArchive,
)
from gepaxmapelite.hashing import CallableDescriptorHasher, DefaultBehaviorDescriptorHasher
from gepaxmapelite.models import (
    AmbiguityStrategy,
    BehaviorDescriptor,
    FitnessReport,
    MetaPromptEvaluation,
    MetaPromptGenome,
    NoiseStrategy,
)


def make_genome(label: str) -> MetaPromptGenome:
    return MetaPromptGenome(
        reflection_template=(
            f"Inspect <curr_param> using the evidence in <side_info>. Variant: {label}"
        ),
        mutation_prompt=f"Produce a useful reflective variation for {label}.",
    )


def make_fitness(
    quality: float,
    *,
    guardrail_violations: tuple[str, ...] = (),
) -> FitnessReport:
    return FitnessReport(
        clean_score=quality,
        ambiguity_score=quality,
        noise_score=quality,
        mixed_score=quality,
        meaning_preservation_score=quality,
        guardrail_violations=guardrail_violations,
    )


def make_evaluation(
    genome: MetaPromptGenome,
    quality: float,
    *,
    cost: float = 1.0,
    variance: float = 0.1,
    guardrail_violations: tuple[str, ...] = (),
) -> MetaPromptEvaluation:
    return MetaPromptEvaluation(
        genome_id=genome.genome_id,
        best_inner_candidate={"task_prompt": f"best prompt from {genome.genome_id[:8]}"},
        fitness=make_fitness(
            quality,
            guardrail_violations=guardrail_violations,
        ),
        total_cost=cost,
        quality_variance=variance,
        inner_run_ids=(f"inner-{genome.genome_id[:8]}",),
    )


def test_selected_inner_run_must_be_present_in_recorded_runs() -> None:
    genome = make_genome("invalid-selected-run")

    with pytest.raises(ValueError, match="must appear in inner_run_ids"):
        MetaPromptEvaluation(
            genome_id=genome.genome_id,
            best_inner_candidate={"task_prompt": "candidate"},
            fitness=make_fitness(0.8),
            best_inner_run_id="ghost",
        )


def test_descriptor_representative_does_not_inherit_an_unrelated_run() -> None:
    genome = make_genome("independent-representative")
    evaluation = MetaPromptEvaluation(
        genome_id=genome.genome_id,
        best_inner_candidate={"task_prompt": "globally best"},
        fitness=make_fitness(0.8),
        inner_run_ids=("best-run", "representative-run"),
        best_inner_run_id="best-run",
        descriptor_representative_candidate={"task_prompt": "cell representative"},
    )

    assert evaluation.descriptor_representative_run_id is None
    assert MetaPromptEvaluation.from_dict(evaluation.to_dict()) == evaluation


def test_distinct_descriptor_run_requires_its_own_candidate() -> None:
    genome = make_genome("missing-representative")

    with pytest.raises(ValueError, match="candidate is required"):
        MetaPromptEvaluation(
            genome_id=genome.genome_id,
            best_inner_candidate={"task_prompt": "globally best"},
            fitness=make_fitness(0.8),
            inner_run_ids=("best-run", "representative-run"),
            best_inner_run_id="best-run",
            descriptor_representative_run_id="representative-run",
        )


def test_legacy_evaluation_defaults_descriptor_evidence_to_best_inner_result() -> None:
    genome = make_genome("legacy-evaluation")
    evaluation = MetaPromptEvaluation(
        genome_id=genome.genome_id,
        best_inner_candidate={"task_prompt": "legacy best"},
        fitness=make_fitness(0.8),
        inner_run_ids=("legacy-run",),
        best_inner_run_id="legacy-run",
    )
    legacy_payload = evaluation.to_dict()
    legacy_payload.pop("descriptor_representative_candidate")
    legacy_payload.pop("descriptor_representative_run_id")

    restored = MetaPromptEvaluation.from_dict(legacy_payload)

    assert restored.descriptor_representative_candidate == restored.best_inner_candidate
    assert restored.descriptor_representative_run_id == restored.best_inner_run_id


def test_inner_run_ids_rejects_one_string_as_a_collection() -> None:
    genome = make_genome("string-run-ids")

    with pytest.raises(TypeError, match="collection of run identifiers"):
        MetaPromptEvaluation(
            genome_id=genome.genome_id,
            best_inner_candidate={"task_prompt": "candidate"},
            fitness=make_fitness(0.8),
            inner_run_ids="run-one",  # type: ignore[arg-type]
        )

    valid = MetaPromptEvaluation(
        genome_id=genome.genome_id,
        best_inner_candidate={"task_prompt": "candidate"},
        fitness=make_fitness(0.8),
    ).to_dict()
    valid["inner_run_ids"] = "run-one"
    with pytest.raises(TypeError, match="collection of run identifiers"):
        MetaPromptEvaluation.from_dict(valid)


def test_default_hasher_covers_all_16_bins_stably() -> None:
    hasher = DefaultBehaviorDescriptorHasher()
    observed: set[str] = set()

    for ambiguity in AmbiguityStrategy:
        for noise in NoiseStrategy:
            descriptor = BehaviorDescriptor(ambiguity=ambiguity, noise=noise)
            expected_index = 4 * int(ambiguity) + int(noise)
            key = hasher.key(descriptor)

            assert key == str(expected_index)
            assert hasher.decode(key) == descriptor
            observed.add(key)

    assert len(list(AmbiguityStrategy)) == 4
    assert len(list(NoiseStrategy)) == 4
    assert observed == {str(index) for index in range(16)}
    assert hasher.possible_keys == tuple(str(index) for index in range(16))


def test_plain_custom_hash_function_can_define_the_bins() -> None:
    hasher = CallableDescriptorHasher(
        lambda descriptor: f"A{int(descriptor.ambiguity)}N{int(descriptor.noise)}",
        version="named-grid-v1",
        possible_keys=tuple(f"A{a}N{n}" for a in range(4) for n in range(4)),
    )
    archive = MapElitesArchive(hasher)
    descriptor = BehaviorDescriptor(AmbiguityStrategy.CLARIFY, NoiseStrategy.VERIFY)
    genome = make_genome("custom-function")

    decision = archive.insert(genome, make_evaluation(genome, 0.8), descriptor)

    assert decision.key == "A3N3"
    assert archive.elite("A3N3") is not None


def test_custom_hash_intentionally_maps_distinct_behaviors_into_one_competition_bin() -> None:
    hasher = CallableDescriptorHasher(
        lambda _descriptor: "shared",
        version="constant-bin-v1",
        possible_keys=("shared",),
    )
    archive = MapElitesArchive(hasher)
    first = make_genome("first behavior")
    second = make_genome("second behavior")

    archive.insert(
        first,
        make_evaluation(first, 0.4),
        BehaviorDescriptor(AmbiguityStrategy.RESOLVE, NoiseStrategy.SURFACE_CLEANUP),
    )
    decision = archive.insert(
        second,
        make_evaluation(second, 0.8),
        BehaviorDescriptor(AmbiguityStrategy.CLARIFY, NoiseStrategy.VERIFY),
    )

    assert decision.reason is AdmissionReason.REPLACED_INCUMBENT
    assert archive.occupied_count == 1
    elite = archive.elite("shared")
    assert elite is not None
    assert elite.genome == second


def test_genome_id_is_canonical_stable_sha256() -> None:
    reflection_template = "Unicode ✓ <curr_param>\nEvidence: <side_info>"
    mutation_prompt = "Mutate deliberately."
    first = MetaPromptGenome(reflection_template, mutation_prompt)
    second = MetaPromptGenome(reflection_template, mutation_prompt)

    canonical = json.dumps(
        {
            "mutation_prompt": mutation_prompt,
            "reflection_template": reflection_template,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert first.genome_id == second.genome_id == expected
    assert first.identity == expected
    assert MetaPromptGenome.from_dict(first.to_dict()) == first
    assert make_genome("different").genome_id != first.genome_id


@pytest.mark.parametrize("missing", ["<curr_param>", "<side_info>"])
def test_genome_rejects_missing_gepa_placeholders(missing: str) -> None:
    template = "Use <curr_param> and <side_info>.".replace(missing, "")
    with pytest.raises(ValueError, match="missing required GEPA placeholder"):
        MetaPromptGenome(template, "Mutate this template.")


def test_archive_keeps_genome_and_its_evaluation_paired_atomically() -> None:
    descriptor = BehaviorDescriptor(AmbiguityStrategy.BRANCH, NoiseStrategy.FILTER)
    archive = MapElitesArchive()
    first = make_genome("first")
    first_evaluation = make_evaluation(first, 0.4)

    initial = archive.insert(first, first_evaluation, descriptor)
    assert initial.accepted
    assert initial.reason is AdmissionReason.FILLED_EMPTY_CELL
    assert initial.elite is not None
    assert initial.elite.genome is first
    assert initial.elite.evaluation is first_evaluation

    before_mismatch = archive.snapshot()
    second = make_genome("second")
    with pytest.raises(ValueError, match="different meta-prompt genome"):
        archive.insert(second, first_evaluation, descriptor)
    assert archive.snapshot() == before_mismatch

    second_evaluation = make_evaluation(second, 0.8)
    replacement = archive.insert(second, second_evaluation, descriptor)
    assert replacement.accepted
    assert replacement.reason is AdmissionReason.REPLACED_INCUMBENT
    assert replacement.displaced is initial.elite

    stored = archive.elite_for(descriptor)
    assert stored is replacement.elite
    assert stored is not None
    assert stored.genome is second
    assert stored.evaluation is second_evaluation
    assert stored.evaluation.genome_id == stored.genome.genome_id


def test_archive_replacement_order_is_quality_then_cost_then_variance_then_id() -> None:
    descriptor = BehaviorDescriptor(AmbiguityStrategy.ASSUME, NoiseStrategy.VERIFY)

    expensive_low_quality = make_genome("expensive-low-quality")
    cheap_high_quality = make_genome("cheap-high-quality")
    archive = MapElitesArchive()
    archive.insert(
        expensive_low_quality,
        make_evaluation(expensive_low_quality, 0.4, cost=1.0, variance=0.1),
        descriptor,
    )
    quality_winner = archive.insert(
        cheap_high_quality,
        make_evaluation(cheap_high_quality, 0.5, cost=100.0, variance=100.0),
        descriptor,
    )
    assert quality_winner.accepted

    lower_cost = make_genome("lower-cost")
    cost_winner = archive.insert(
        lower_cost,
        make_evaluation(lower_cost, 0.5, cost=99.0, variance=200.0),
        descriptor,
    )
    assert cost_winner.accepted
    assert archive.elite_for(descriptor).genome == lower_cost  # type: ignore[union-attr]

    lower_variance = make_genome("lower-variance")
    variance_winner = archive.insert(
        lower_variance,
        make_evaluation(lower_variance, 0.5, cost=99.0, variance=1.0),
        descriptor,
    )
    assert variance_winner.accepted
    assert archive.elite_for(descriptor).genome == lower_variance  # type: ignore[union-attr]

    candidates = sorted(
        (make_genome("identity-a"), make_genome("identity-b")),
        key=lambda genome: genome.genome_id,
    )
    lower_id, higher_id = candidates
    id_archive = MapElitesArchive()
    id_archive.insert(
        higher_id,
        make_evaluation(higher_id, 0.6, cost=1.0, variance=0.0),
        descriptor,
    )
    id_winner = id_archive.insert(
        lower_id,
        make_evaluation(lower_id, 0.6, cost=1.0, variance=0.0),
        descriptor,
    )
    assert id_winner.accepted
    assert id_archive.elite_for(descriptor).genome == lower_id  # type: ignore[union-attr]


def test_guardrail_violation_is_rejected_without_filling_cell() -> None:
    archive = MapElitesArchive()
    genome = make_genome("unsafe")
    evaluation = make_evaluation(
        genome,
        1.0,
        guardrail_violations=("invented fact",),
    )
    descriptor = BehaviorDescriptor(AmbiguityStrategy.CLARIFY, NoiseStrategy.VERIFY)

    decision = archive.insert(genome, evaluation, descriptor)

    assert not decision.accepted
    assert decision.reason is AdmissionReason.GUARDRAIL_VIOLATION
    assert decision.elite is None
    assert archive.elite_for(descriptor) is None
    assert len(archive) == 0


class CustomDescriptorHasher:
    version = "custom-grid-v1"
    possible_keys: Collection[str] = tuple(
        f"custom/{ambiguity}/{noise}" for ambiguity in range(4) for noise in range(4)
    )

    def key(self, descriptor: BehaviorDescriptor) -> str:
        return f"custom/{int(descriptor.ambiguity)}/{int(descriptor.noise)}"


class IncompatibleCustomDescriptorHasher(CustomDescriptorHasher):
    version = "custom-grid-v2"


def test_custom_hasher_key_and_version_are_enforced() -> None:
    hasher = CustomDescriptorHasher()
    archive = MapElitesArchive(hasher=hasher)
    genome = make_genome("custom")
    descriptor = BehaviorDescriptor(AmbiguityStrategy.CLARIFY, NoiseStrategy.EXTRACT)

    decision = archive.insert(genome, make_evaluation(genome, 0.7), descriptor)
    snapshot = archive.snapshot()

    assert decision.key == "custom/3/2"
    assert tuple(snapshot["elites"]) == ("custom/3/2",)
    assert snapshot["hasher_version"] == "custom-grid-v1"
    assert MapElitesArchive.from_snapshot(snapshot, hasher=hasher).snapshot() == snapshot

    with pytest.raises(HasherVersionMismatchError):
        MapElitesArchive.from_snapshot(snapshot)
    with pytest.raises(HasherVersionMismatchError):
        MapElitesArchive.from_snapshot(
            snapshot,
            hasher=IncompatibleCustomDescriptorHasher(),
        )


def test_json_save_load_round_trip(tmp_path) -> None:
    archive = MapElitesArchive()
    first = make_genome("json-one")
    second = make_genome("json-two")
    archive.insert(
        first,
        make_evaluation(first, 0.55, cost=2.0, variance=0.03),
        BehaviorDescriptor(AmbiguityStrategy.RESOLVE, NoiseStrategy.SURFACE_CLEANUP),
    )
    archive.insert(
        second,
        make_evaluation(second, 0.75, cost=3.0, variance=0.02),
        BehaviorDescriptor(AmbiguityStrategy.CLARIFY, NoiseStrategy.VERIFY),
    )

    path = tmp_path / "nested" / "archive.json"
    archive.save(path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    restored = MapElitesArchive.load(path)

    assert parsed == archive.snapshot()
    assert restored.snapshot() == archive.snapshot()
    assert restored.elite("0").genome == first  # type: ignore[union-attr]
    assert restored.elite("15").evaluation == make_evaluation(  # type: ignore[union-attr]
        second,
        0.75,
        cost=3.0,
        variance=0.02,
    )
    assert not list(path.parent.glob("*.tmp"))
