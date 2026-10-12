"""Evaluation integrity checks; these do not measure real-video accuracy."""

from copy import deepcopy

import pytest

from landmark_stabilizer.wobble import wobble


FEATURE = "deformation_p95"


@pytest.fixture
def labeled_pairs():
    examples = []
    pairs = []
    for index in range(4):
        source = f"recording-{index}"
        for role, label, value in (
            ("before", "too_wobbly", 8.0 + index / 10),
            ("after", "not_too_wobbly", 0.1 + index / 100),
        ):
            examples.append(
                {
                    "id": f"{source}-{role}",
                    "source_id": source,
                    "label": label,
                    "features": {FEATURE: value},
                    "sha256": f"{len(examples) + 1:064x}",
                }
            )
        pairs.append({"before_id": f"{source}-before", "after_id": f"{source}-after"})
    return examples, pairs


def evaluate(examples, pairs):
    return wobble.evaluate_examples(examples, pairs, feature_names=[FEATURE])


def test_separable_fixture_passes_each_absolute_label_and_each_pair(labeled_pairs):
    examples, pairs = labeled_pairs
    result = evaluate(examples, pairs)
    for mode in ("resubstitution", "group_holdout"):
        metrics = result[mode]
        assert metrics["strict_accuracy"] == 1.0
        assert metrics["pairs"]["total"] == metrics["pairs"]["correct"] == 4
        assert metrics["confusion"] == {"tp": 4, "tn": 4, "fp": 0, "fn": 0, "abstain": 0}


def test_fold_excludes_every_variant_of_the_original_recording(labeled_pairs):
    examples, pairs = labeled_pairs
    result = evaluate(examples, pairs)
    predictions = result["group_holdout"]["predictions"]
    assert {row["id"] for row in predictions} == {row["id"] for row in examples}
    for row in predictions:
        assert row["source_id"] not in row["training_source_ids"]
        assert set(row["training_source_ids"]) == {
            example["source_id"] for example in examples if example["source_id"] != row["source_id"]
        }
    by_id = {row["id"]: row for row in predictions}
    for pair in pairs:
        before, after = by_id[pair["before_id"]], by_id[pair["after_id"]]
        assert before["training_source_ids"] == after["training_source_ids"]


def test_better_rank_does_not_make_two_rejected_predictions_a_passing_pair(
    labeled_pairs, monkeypatch
):
    examples, pairs = labeled_pairs
    monkeypatch.setattr(
        wobble,
        "predict_features",
        lambda features, model: {"prediction": "too_wobbly", "score": features[FEATURE]},
    )
    result = evaluate(examples, pairs)
    for mode in ("resubstitution", "group_holdout"):
        assert result[mode]["pairs"]["correct"] == 0
        assert result[mode]["strict_accuracy"] == 0.5
        assert result[mode]["confusion"]["fp"] == 4


def test_abstentions_remain_in_strict_accuracy_and_pair_denominators(labeled_pairs, monkeypatch):
    examples, pairs = labeled_pairs
    monkeypatch.setattr(
        wobble,
        "predict_features",
        lambda features, model: {"prediction": "needs_review", "score": None},
    )
    result = evaluate(examples, pairs)
    for mode in ("resubstitution", "group_holdout"):
        assert result[mode]["strict_accuracy"] == 0.0
        assert result[mode]["pairs"]["total"] == 4
        assert result[mode]["pairs"]["correct"] == 0
        assert result[mode]["confusion"]["abstain"] == 8


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_unmeasured_or_nonfinite_required_feature_cannot_pass(labeled_pairs, value):
    examples, _ = labeled_pairs
    model = wobble.fit_model(examples, feature_names=[FEATURE])
    prediction = wobble.predict_features({FEATURE: value}, model)
    assert prediction["prediction"] == "needs_review"


def test_missing_required_feature_cannot_be_imputed_into_a_pass(labeled_pairs):
    examples, _ = labeled_pairs
    model = wobble.fit_model(examples, feature_names=[FEATURE])
    assert wobble.predict_features({}, model)["prediction"] == "needs_review"


def test_identical_model_and_predictions_with_fixed_inputs(labeled_pairs):
    examples, pairs = labeled_pairs
    assert evaluate(examples, pairs) == evaluate(examples, pairs)


def test_non_feature_identity_and_review_metadata_do_not_change_predictions(labeled_pairs):
    examples, _ = labeled_pairs
    model = wobble.fit_model(examples, feature_names=[FEATURE])
    renamed = deepcopy(examples)
    for index, row in enumerate(renamed):
        row.update(
            id=f"unrelated-{index}",
            source_id=f"unrelated-source-{index // 2}",
            filename=f"approved-repaired-{index}.mp4",
            approved=row["label"] == "too_wobbly",
            repair_attempt=1000 - index,
        )
        row["features"]["approved"] = float(row["approved"])
        row["features"]["repair_attempt"] = float(row["repair_attempt"])
    renamed_model = wobble.fit_model(renamed, feature_names=[FEATURE])
    for features in ({FEATURE: 0.15}, {FEATURE: 8.2}):
        assert wobble.predict_features(features, model) == wobble.predict_features(features, renamed_model)


def test_conflicting_labels_for_same_export_hash_are_rejected(labeled_pairs):
    examples, _ = labeled_pairs
    contradictory = deepcopy(examples[0])
    contradictory.update(id="contradiction", label="not_too_wobbly")
    with pytest.raises(ValueError):
        wobble.fit_model(examples + [contradictory], feature_names=[FEATURE])


def test_duplicate_example_ids_cannot_hide_conflicting_labels(labeled_pairs):
    examples, _ = labeled_pairs
    contradictory = deepcopy(examples[0])
    contradictory.update(label="not_too_wobbly", sha256="a" * 64)
    with pytest.raises(ValueError):
        wobble.fit_model(examples + [contradictory], feature_names=[FEATURE])


@pytest.mark.parametrize(
    "pair",
    [
        {"before_id": "missing", "after_id": "recording-0-after"},
        {"before_id": "recording-0-before", "after_id": "recording-1-after"},
        {"before_id": "recording-0-after", "after_id": "recording-0-before"},
    ],
)
def test_invalid_or_reversed_pairs_are_rejected(labeled_pairs, pair):
    examples, _ = labeled_pairs
    with pytest.raises(ValueError):
        evaluate(examples, [pair])


def test_single_source_cannot_claim_generalization_to_unseen_recordings(labeled_pairs):
    examples, pairs = labeled_pairs
    result = evaluate(examples[:2], pairs[:1])
    held_out = result["group_holdout"]
    assert held_out["strict_accuracy"] == 0.0
    assert held_out["confusion"]["abstain"] == 2
    assert held_out["pairs"]["correct"] == 0


@pytest.mark.parametrize("value", [True, False, "0.1", [0.1], [[0.1]]])
def test_required_motion_feature_must_be_a_real_scalar(labeled_pairs, value):
    examples, _ = labeled_pairs
    model = wobble.fit_model(examples, feature_names=[FEATURE])
    assert wobble.predict_features({FEATURE: value}, model)["prediction"] == "needs_review"


@pytest.mark.parametrize(
    "change",
    [
        {"trees": []},
        {"trees": [{"value": float("nan"), "n": 1}]},
        {"threshold": float("nan")},
        {"threshold": 2.0},
        {"model_version": "unknown-future-model"},
        {"schema_version": 999},
    ],
)
def test_malformed_model_cannot_produce_an_acceptable_verdict(labeled_pairs, change):
    examples, _ = labeled_pairs
    model = wobble.fit_model(examples, feature_names=[FEATURE])
    model.update(change)
    with pytest.raises(ValueError):
        wobble.predict_features({FEATURE: 0.1}, model)


def test_unusable_measurement_cannot_train_or_pass_through_truthy_status(labeled_pairs):
    examples, pairs = labeled_pairs
    examples[0]["features_available"] = "false"
    result = evaluate(examples, pairs)
    prediction = next(row for row in result["resubstitution"]["predictions"] if row["id"] == examples[0]["id"])
    assert prediction["prediction"] == "needs_review"
    assert examples[0]["id"] not in result["model"]["training_example_ids"]


def test_shared_split_group_keeps_source_aliases_out_of_training(labeled_pairs):
    examples, pairs = labeled_pairs
    for row in examples:
        row["split_group"] = "same-capture" if row["source_id"] in {"recording-0", "recording-1"} else row["source_id"]
    result = evaluate(examples, pairs)
    for row in result["group_holdout"]["predictions"]:
        if row["source_id"] in {"recording-0", "recording-1"}:
            assert not set(row["training_source_ids"]) & {"recording-0", "recording-1"}


def test_pair_categories_remain_auditable(labeled_pairs):
    examples, pairs = labeled_pairs
    pairs[0]["repair_kind"] = "overlapping_source_interval_recut"
    pairs[1]["repair_kind"] = "same_interval_repair"
    result = evaluate(examples, pairs)
    for mode in ("resubstitution", "group_holdout"):
        details = result[mode]["pairs"]["details"]
        assert details[0]["repair_kind"] == pairs[0]["repair_kind"]
        assert details[1]["repair_kind"] == pairs[1]["repair_kind"]


def test_variants_of_one_source_cannot_assign_themselves_different_split_groups(labeled_pairs):
    examples, pairs = labeled_pairs
    examples[0]["split_group"] = "before-only-group"
    with pytest.raises(ValueError):
        evaluate(examples, pairs)
