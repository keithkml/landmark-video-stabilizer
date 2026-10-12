"""The CLI must bind motion measurements to exact bytes and fail closed."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from landmark_stabilizer.wobble import cli
from landmark_stabilizer.wobble import wobble_features


def measured_analysis():
    aggregate = {"valid_pair_fraction": 1.0}
    for name in wobble_features.FEATURE_NAMES:
        for suffix in ("median", "p90", "p95"):
            aggregate[f"{name}_{suffix}"] = 0.1
    for name in ("deformation", "row"):
        aggregate[f"{name}_oscillation_px"] = 0.1
        aggregate[f"{name}_reversal_fraction"] = 0.1
    for name in ("scale", "rotation"):
        aggregate[f"{name}_pumping_px"] = 0.1
        aggregate[f"{name}_pumping_reversal_fraction"] = 0.1
    aggregate.update(coverage_median=1.0, row_coverage_median=1.0, track_count_median=100.0)
    for name, value in list(aggregate.items()):
        if name != "valid_pair_fraction":
            aggregate[f"worst_window_{name}"] = value
    return {
        "schema_version": 1,
        "extractor_version": wobble_features.EXTRACTOR_VERSION,
        "quality": {"usable": True, "reasons": []},
        "config": {
            "sample_fps": 12.0,
            "width": 640,
            "window_seconds": 2.0,
            "stride_seconds": 1.0,
            "reference_width": 640.0,
            "reference_fps": 12.0,
            "grid_columns": 8,
            "grid_rows": 6,
            "max_seconds": None,
        },
        "sampling": {"sampled_frames": 25, "pair_count": 24, "analyzed_duration_seconds": 2.0},
        "aggregate": aggregate,
        "windows": [],
    }


@pytest.mark.parametrize("quality", [None, {}, {"usable": "true"}, {"usable": "false"}, {"usable": 1}])
def test_missing_or_invalid_quality_cannot_make_measurements_usable(quality):
    analysis = measured_analysis()
    if quality is None:
        analysis.pop("quality")
    else:
        analysis["quality"] = quality
    assert cli.is_usable(analysis) is False


def test_quality_failure_and_nonfinite_required_features_cannot_pass():
    analysis = measured_analysis()
    assert cli.is_usable(analysis) is True
    required = sorted(cli.numeric_features(analysis))[0]
    analysis["aggregate"][required] = float("nan")
    assert cli.is_usable(analysis) is False
    analysis = measured_analysis()
    analysis["quality"] = {"usable": False, "reasons": ["insufficient_spatial_coverage"]}
    assert cli.is_usable(analysis) is False


def test_numeric_metadata_does_not_enter_the_production_feature_vector():
    analysis = measured_analysis()
    analysis["aggregate"].update(approved=1, label=0, repair_attempt=2, source_id=42)
    features = cli.numeric_features(analysis)
    assert not {"approved", "label", "repair_attempt", "source_id"} & set(features)
    assert features


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    examples, pairs = [], []
    for source in ("one", "two"):
        for role, label in (("before", "too_wobbly"), ("after", "not_too_wobbly")):
            video = tmp_path / f"{source}-{role}.mp4"
            video.write_bytes(f"fake video for provenance tests: {source} {role}".encode())
            examples.append({
                "id": f"{source}-{role}", "source_id": source, "split_group": source,
                "label": label, "sha256": cli.digest(video), "video_path": str(video),
                "eligible_for_primary_eval": True,
            })
        pairs.append({"before_id": f"{source}-before", "after_id": f"{source}-after",
                      "repair_kind": "same_interval_repair", "eligible_for_primary_eval": True})
    manifest = tmp_path / "manifest.json"
    cli.write(manifest, {"schema": "user-wobble-eval-v1", "examples": examples, "pairs": pairs})
    calls = []

    def fake_analyze(video, **config):
        calls.append((video, config))
        return measured_analysis()

    monkeypatch.setattr(wobble_features, "analyze_video", fake_analyze)
    monkeypatch.setattr(cli, "signature", lambda: "a" * 64)
    args = SimpleNamespace(manifest=str(manifest), output=str(tmp_path / "features"), sample_fps=12.0, width=640)
    return args, examples, calls


def evaluation_args(args):
    return SimpleNamespace(manifest=args.manifest, features=str(Path(args.output) / "extracted.json"),
                           output=str(Path(args.output).parent / "model"))


def test_extract_rejects_wrong_media_bytes_without_analyzing_them(corpus):
    args, examples, calls = corpus
    Path(examples[0]["video_path"]).write_bytes(b"changed after the user review")
    cli.extract(args)
    result = cli.read(Path(args.output) / "extracted.json")
    row = result["examples"][0]
    assert row["features_available"] is False
    assert "match" in row["feature_error"].lower()
    assert not any(video == examples[0]["video_path"] for video, _ in calls)


@pytest.mark.parametrize("field", ["media_sha256", "extractor_sha256", "extraction_config"])
def test_swapped_or_stale_cache_is_never_accepted_as_current_evidence(corpus, field):
    args, _, calls = corpus
    cli.extract(args)
    extracted = cli.read(Path(args.output) / "extracted.json")
    cache_path = Path(extracted["examples"][0]["analysis_path"])
    cache = cli.read(cache_path)
    cache[field] = {"sample_fps": 1.0} if field == "extraction_config" else "b" * 64
    cli.write(cache_path, cache)
    previous_calls = len(calls)
    cli.extract(args)
    current = cli.read(Path(args.output) / "extracted.json")["examples"][0]
    assert len(calls) > previous_calls or current["features_available"] is False


def test_valid_cache_is_reused_and_evaluation_preserves_pair_category(corpus):
    args, _, calls = corpus
    cli.extract(args)
    previous_calls = len(calls)
    cli.extract(args)
    assert len(calls) == previous_calls
    evaluate_args = evaluation_args(args)
    cli.evaluate(evaluate_args)
    result = cli.read(Path(evaluate_args.output) / "evaluation.json")
    assert result["acceptance_gate"]["automatic_approval_permitted"] is False
    for mode in ("resubstitution", "group_holdout"):
        assert all(pair["repair_kind"] == "same_interval_repair" for pair in result[mode]["pairs"]["details"])


@pytest.mark.parametrize("change", ["corpus_hash", "extractor_hash", "derived_value", "quality_flag", "missing_row", "split_group", "complete_flag"])
def test_evaluation_rejects_unbound_or_modified_extracted_evidence(corpus, change):
    args, _, _ = corpus
    cli.extract(args)
    path = Path(args.output) / "extracted.json"
    extracted = cli.read(path)
    if change == "corpus_hash":
        extracted["manifest_sha256"] = "c" * 64
    elif change == "extractor_hash":
        extracted["extractor_sha256"] = "c" * 64
    elif change == "derived_value":
        key = sorted(extracted["examples"][0]["features"])[0]
        extracted["examples"][0]["features"][key] += 99
    elif change == "quality_flag":
        extracted["examples"][0]["features_available"] = False
    elif change == "missing_row":
        extracted["examples"].pop()
    elif change == "split_group":
        extracted["examples"][0]["split_group"] = "different-group"
    elif change == "complete_flag":
        extracted["complete"] = "false"
    cli.write(path, extracted)
    with pytest.raises(ValueError):
        cli.evaluate(evaluation_args(args))


def test_eligible_pair_with_missing_endpoint_cannot_disappear_from_the_evaluation(corpus):
    args, _, _ = corpus
    manifest = cli.read(args.manifest)
    manifest["pairs"][0]["before_id"] = "missing-export"
    cli.write(args.manifest, manifest)
    cli.extract(args)
    with pytest.raises(ValueError):
        cli.evaluate(evaluation_args(args))


def test_detection_is_advisory_and_insufficient_quality_abstains(corpus, monkeypatch, tmp_path):
    args, examples, _ = corpus
    cli.extract(args)
    eval_args = evaluation_args(args)
    cli.evaluate(eval_args)
    analysis = measured_analysis()
    analysis["quality"] = {"usable": False, "reasons": ["incomplete_decode"]}
    monkeypatch.setattr(wobble_features, "analyze_video", lambda *args, **kwargs: deepcopy(analysis))
    detect_args = SimpleNamespace(model=str(Path(eval_args.output) / "model.json"),
                                  video=examples[0]["video_path"], output=str(tmp_path / "decision.json"))
    cli.detect(detect_args)
    decision = cli.read(detect_args.output)
    assert decision["prediction"] == "needs_review"
    assert decision["automatic_approval_permitted"] is False
    assert decision["sha256"] == examples[0]["sha256"]


@pytest.fixture
def overlap_corpus(corpus, monkeypatch):
    """Two source passages shifted by one second exercise source/export mapping."""
    args, examples, calls = corpus
    manifest = cli.read(args.manifest)
    for example in manifest["examples"]:
        start = 5.0 if example["id"].endswith("before") else 6.0
        example["source_interval"] = {"start_sec": start, "end_sec": start + 2.0}
    for index, pair in enumerate(manifest["pairs"]):
        pair["id"] = f"pair-{index}"
        pair["repair_kind"] = "overlapping_source_interval_recut"
    cli.write(args.manifest, manifest)

    def fake_analyze(video, **config):
        calls.append((video, config))
        analysis = measured_analysis()
        analysis["sampling"].update(source_duration_seconds=2.0, source_fps=12.0, source_frame_count=24)
        return analysis

    monkeypatch.setattr(wobble_features, "analyze_video", fake_analyze)
    cli.extract(args)
    overlap_args = SimpleNamespace(manifest=args.manifest, features=str(Path(args.output) / "extracted.json"),
                                   output=str(Path(args.output) / "overlap.json"))
    calls.clear()
    return overlap_args, manifest, calls


def test_overlap_maps_each_export_to_shared_source_time_and_adds_no_labels(overlap_corpus):
    args, manifest, calls = overlap_corpus
    cli.overlap(args)
    output = cli.read(args.output)
    assert output["complete"] is True
    assert "no separate user labels" in output["purpose"]
    assert len(output["pairs"]) == len(manifest["pairs"]) == 2
    for pair in output["pairs"]:
        assert pair["source_interval_seconds"] == [6.0, 7.0]
        assert pair["before"]["export_bounds_seconds"] == {"start_seconds": 1.0, "end_seconds": 2.0}
        assert pair["after"]["export_bounds_seconds"] == {"start_seconds": 0.0, "end_seconds": 1.0}
        assert not pair["before"]["reused_full_export"]
        assert not pair["after"]["reused_full_export"]
        assert "label" not in pair["before"] and "label" not in pair["after"]
    assert len(calls) == 4
    cli.overlap(args)
    assert len(calls) == 4, "Repeated diagnostics should reuse bound segment measurements"


def test_overlap_reuses_full_export_only_when_all_source_frames_are_included(overlap_corpus):
    args, manifest, calls = overlap_corpus
    for example in manifest["examples"]:
        example["source_interval"] = {"start_sec": 5.0, "end_sec": 7.0}
    cli.write(args.manifest, manifest)
    extracted = cli.read(args.features)
    extracted["manifest_sha256"] = cli.digest(args.manifest)
    cli.write(args.features, extracted)
    cli.overlap(args)
    assert not calls
    assert all(p[role]["reused_full_export"] for p in cli.read(args.output)["pairs"] for role in ("before", "after"))

    # Two-frame metadata tolerance must not allow reuse after removing the last
    # measurable frame: the requested end is before timestamp 23/12 seconds.
    manifest["examples"][1]["source_interval"]["end_sec"] = 6.9
    cli.write(args.manifest, manifest)
    extracted["manifest_sha256"] = cli.digest(args.manifest)
    cli.write(args.features, extracted)
    cli.overlap(args)
    first = cli.read(args.output)["pairs"][0]
    assert not first["before"]["reused_full_export"]
    assert not first["after"]["reused_full_export"]
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["media_bytes", "extractor_hash", "complete_flag", "manifest_hash", "cache_bounds"])
def test_overlap_rejects_changed_media_or_unbound_measurements(overlap_corpus, change):
    args, manifest, calls = overlap_corpus
    if change == "media_bytes":
        Path(manifest["examples"][0]["video_path"]).write_bytes(b"different reviewed video")
    elif change == "cache_bounds":
        cli.overlap(args)
        record = cli.read(args.output)["pairs"][0]["before"]
        cache = cli.read(record["analysis_path"])
        cache["extraction_config"]["start_seconds"] = 0.0
        cli.write(record["analysis_path"], cache)
    else:
        extracted = cli.read(args.features)
        field = {"extractor_hash": "extractor_sha256", "complete_flag": "complete", "manifest_hash": "manifest_sha256"}[change]
        extracted[field] = "false" if change == "complete_flag" else "b" * 64
        cli.write(args.features, extracted)
    calls.clear()
    with pytest.raises(ValueError):
        cli.overlap(args)
    assert not calls


@pytest.mark.parametrize("change", ["negative", "reversed", "boolean", "no_overlap", "different_source", "duration_mismatch"])
def test_overlap_refuses_invalid_or_unmappable_source_passages(overlap_corpus, change):
    args, manifest, calls = overlap_corpus
    before, after = manifest["examples"][:2]
    if change == "negative":
        before["source_interval"]["start_sec"] = -1.0
    elif change == "reversed":
        before["source_interval"] = {"start_sec": 7.0, "end_sec": 5.0}
    elif change == "boolean":
        before["source_interval"]["start_sec"] = True
    elif change == "no_overlap":
        after["source_interval"] = {"start_sec": 7.0, "end_sec": 9.0}
    elif change == "different_source":
        after["source_id"] = "different-source"
    elif change == "duration_mismatch":
        before["source_interval"]["end_sec"] = 7.5
    cli.write(args.manifest, manifest)
    extracted = cli.read(args.features)
    extracted["manifest_sha256"] = cli.digest(args.manifest)
    cli.write(args.features, extracted)
    with pytest.raises(ValueError):
        cli.overlap(args)
    assert not calls


@pytest.mark.parametrize("field,value", [("source_fps", 0), ("source_duration_seconds", True), ("source_frame_count", 1.5)])
def test_overlap_rejects_invalid_source_timing_metadata(overlap_corpus, field, value):
    args, _, calls = overlap_corpus
    extracted = cli.read(args.features)
    cache_path = extracted["examples"][0]["analysis_path"]
    analysis = cli.read(cache_path)
    analysis["sampling"][field] = value
    cli.write(cache_path, analysis)
    with pytest.raises(ValueError):
        cli.overlap(args)
    assert not calls
