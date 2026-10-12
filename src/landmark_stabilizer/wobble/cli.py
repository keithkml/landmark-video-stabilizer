#!/usr/bin/env python3
"""Extract motion evidence, evaluate user preferences, and score a new export.

Use wobble-detector --help or landmark-stabilizer wobble VIDEO.
This tool never writes media, review state, or approval databases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .storage import media_filename
from .wobble import PREDICTION_FEATURES, evaluate_examples, predict_features, validate_model


DEFAULT_MODEL = Path(__file__).with_name("data") / "native-v3.json"

def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def eligible(manifest):
    return [e for e in manifest["examples"] if e.get("eligible_for_primary_eval", True)]


def signature():
    from . import wobble_features
    return digest(wobble_features.__file__)


def predictor_signature():
    from . import wobble
    return digest(wobble.__file__)


def numeric_features(analysis):
    values = analysis.get("aggregate", {})
    return {k: values[k] for k in PREDICTION_FEATURES if k in values
            and isinstance(values[k], (float, int)) and not isinstance(values[k], bool)
            and math.isfinite(values[k])}


def is_usable(analysis):
    quality = analysis.get("quality", {})
    if quality.get("usable") is not True or quality.get("reasons") or quality.get("status") in {
        "insufficient_texture", "insufficient_coverage", "insufficient_motion_support",
        "incomplete_decode", "insufficient_frames", "error"
    }:
        return False
    return len(numeric_features(analysis)) == len(PREDICTION_FEATURES)


def verify_analysis(analysis, sha, extractor_sha, config):
    if (analysis.get("media_sha256") != sha or analysis.get("extractor_sha256") != extractor_sha
            or analysis.get("extraction_config") != config):
        raise ValueError("Cached analysis provenance does not match media, extractor, and configuration")


def extract(args):
    from .wobble_features import analyze_video
    manifest = read(args.manifest)
    corpus_sha = digest(args.manifest)
    config = {"sample_fps": args.sample_fps, "width": args.width,
              "window_seconds": 2.0, "stride_seconds": 1.0}
    extractor_sha = signature()
    rows, examples = [], eligible(manifest)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for index, e in enumerate(examples, 1):
        started = time.monotonic()
        row = {"id": e["id"], "source_id": e["source_id"], "label": e["label"],
               "split_group": e.get("split_group", e["source_id"]),
               "sha256": e["sha256"], "features": {}, "features_available": False}
        try:
            path = Path(media_filename(e["video_path"]))
            before = path.stat()
            actual = digest(path)
            if actual != e["sha256"]:
                raise ValueError(f"Video bytes no longer match the frozen labeled export ({actual})")
            key = hashlib.sha256(json.dumps([actual, extractor_sha, config], sort_keys=True).encode()).hexdigest()
            cache = out / "features" / (key + ".json")
            if cache.exists():
                analysis = read(cache)
                verify_analysis(analysis, actual, extractor_sha, config)
            else:
                analysis = analyze_video(str(path), **config)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Video changed during analysis; its evidence cannot be used")
                analysis.update(media_sha256=actual, extractor_sha256=extractor_sha, extraction_config=config)
                write(cache, analysis)
            row.update(features=numeric_features(analysis), features_available=is_usable(analysis),
                       quality=analysis.get("quality"), analysis_path=str(cache.resolve()))
            if not row["features_available"]:
                row["feature_error"] = "Motion extraction has insufficient support: " + str(analysis.get("quality"))
        except Exception as exc:
            row["feature_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write(out / "extracted.json", {"schema_version": 1, "manifest_sha256": corpus_sha,
              "extractor_sha256": extractor_sha, "extraction_config": config,
              "complete": len(rows) == len(examples), "expected_examples": len(examples), "examples": rows})
        print(json.dumps({"index": index, "total": len(examples), "id": e["id"],
                          "usable": row["features_available"], "seconds": round(time.monotonic()-started, 2),
                          "error": row.get("feature_error")}), flush=True)


def evaluate(args):
    manifest, extracted = read(args.manifest), read(args.features)
    if extracted["manifest_sha256"] != digest(args.manifest):
        raise ValueError("The corpus changed after extraction. Extract the frozen corpus again.")
    if extracted.get("complete") is not True:
        raise ValueError("Feature extraction is incomplete")
    if extracted.get("extractor_sha256") != signature():
        raise ValueError("Extractor changed after extraction; extract again before evaluating")
    examples = eligible(manifest)
    expected = {e["id"]: e for e in examples}
    rows = extracted["examples"]
    if {e["id"] for e in rows} != set(expected) or len(rows) != len(expected):
        raise ValueError("Feature rows do not exactly cover the eligible frozen corpus")
    for row in rows:
        if any(row[key] != expected[row["id"]][key] for key in ("label", "sha256", "source_id")):
            raise ValueError("Feature provenance and corpus disagree")
        if row.get("split_group", row["source_id"]) != expected[row["id"]].get("split_group", row["source_id"]):
            raise ValueError("Feature split group and corpus disagree")
        if row.get("analysis_path"):
            analysis = read(row["analysis_path"])
            verify_analysis(analysis, row["sha256"], extracted["extractor_sha256"], extracted["extraction_config"])
            if row["features"] != numeric_features(analysis) or row["features_available"] != is_usable(analysis):
                raise ValueError("Feature row differs from its verified analysis artifact")
        elif row.get("features_available") or row.get("features"):
            raise ValueError("Usable feature row is missing its analysis artifact")
    pairs = [p for p in manifest.get("pairs", []) if p.get("eligible_for_primary_eval", True)]
    if any(p["before_id"] not in expected or p["after_id"] not in expected for p in pairs):
        raise ValueError("Eligible pair references an unknown or ineligible example")
    result = evaluate_examples(rows, pairs, feature_names=PREDICTION_FEATURES)
    result.update(manifest_sha256=extracted["manifest_sha256"], extractor_sha256=extracted["extractor_sha256"],
                  predictor_sha256=predictor_signature(), extraction_config=extracted["extraction_config"],
                  corpus_path=str(Path(args.manifest).resolve()),
                  evaluation_context="Development cross-validation: all variants of the test source are excluded from fitting. This corpus has informed detector development; it is not an untouched prospective test.")
    import cv2
    import numpy as np
    result["runtime"] = {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__}
    model = result.pop("model")
    model.update(extraction_config=extracted["extraction_config"], extractor_sha256=extracted["extractor_sha256"],
                 predictor_sha256=result["predictor_sha256"], runtime=result["runtime"],
                 evaluation_context=result["evaluation_context"],
                 manifest_sha256=extracted["manifest_sha256"], validation=result["acceptance_gate"])
    write(Path(args.output) / "model.json", model)
    write(Path(args.output) / "evaluation.json", result)
    from . import wobble, wobble_features
    for module, name in ((wobble, "wobble.py"), (wobble_features, "wobble_features.py")):
        source = Path(module.__file__).read_text()
        target = Path(args.output) / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    print(json.dumps({key: {k: v for k, v in result[key].items() if k not in ("predictions", "folds")}
                      for key in ("resubstitution", "group_holdout", "acceptance_gate")}, indent=2))


def detect(args):
    from .wobble_features import analyze_video
    # Refuse an existing destination before decoding, including links to input.
    output = Path(args.output).expanduser() if args.output else None
    if output is not None and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"Output already exists; choose a new file: {output}")
    model = read(args.model)
    validate_model(model)
    if set(model["feature_names"]) != set(PREDICTION_FEATURES):
        raise ValueError("Model does not use the supported motion-feature schema")
    if model.get("extractor_sha256") != signature():
        raise ValueError("Extractor code differs from the calibrated model; re-extract and evaluate before use")
    if model.get("predictor_sha256") != predictor_signature():
        raise ValueError("Predictor code differs from the evaluated model; evaluate again before use")
    path = Path(media_filename(args.video))
    before = path.stat()
    sha = digest(path)
    analysis = analyze_video(str(path), **model["extraction_config"])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Video changed during analysis")
    decision = predict_features(numeric_features(analysis), model) if is_usable(analysis) else {
        "prediction": "needs_review", "score": None, "reason": "Insufficient measurable motion support"}
    result = {"video_path": str(path.resolve()), "sha256": sha, **decision,
              "model_sha256": digest(args.model),
              "model_version": model["model_version"], "model_validation": model["validation"],
              "evaluation_context": model.get("evaluation_context"),
              "automatic_approval_permitted": False, "analysis": analysis}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against a destination appearing while
        # analysis runs. Never replace media, a model, or a previous result.
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "analysis"}, indent=2))


def overlap(args):
    """Measure shared source time without treating subclips as new user labels."""
    from .wobble_features import analyze_video
    manifest, extracted = read(args.manifest), read(args.features)
    if extracted.get("complete") is not True or extracted["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Overlap diagnostics require a completed matching full-export extraction")
    if extracted["extractor_sha256"] != signature():
        raise ValueError("Extractor changed since the full-export extraction")
    examples = {e["id"]: e for e in eligible(manifest)}
    full = {r["id"]: r for r in extracted["examples"]}
    rows, config = [], extracted["extraction_config"]
    pairs = [p for p in manifest["pairs"] if p.get("eligible_for_primary_eval", True)]
    for pair in pairs:
        before, after = examples[pair["before_id"]], examples[pair["after_id"]]
        if before["source_id"] != after["source_id"]:
            raise ValueError("Shared-passage pair must belong to the same source recording")
        for e in (before, after):
            interval = e["source_interval"]
            values = [interval["start_sec"], interval["end_sec"]]
            if (any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values)
                    or not 0 <= values[0] < values[1]):
                raise ValueError("Source interval must have finite increasing nonnegative bounds")
        start = max(e["source_interval"]["start_sec"] for e in (before, after))
        end = min(e["source_interval"]["end_sec"] for e in (before, after))
        if end <= start:
            raise ValueError("An eligible pair has no shared source time")
        record = dict(id=pair["id"], source_interval_seconds=[start, end], repair_kind=pair["repair_kind"])
        for role, e in (("before", before), ("after", after)):
            a = read(full[e["id"]]["analysis_path"])
            verify_analysis(a, e["sha256"], extracted["extractor_sha256"], config)
            path = Path(media_filename(e["video_path"]))
            stat_before = path.stat()
            if digest(path) != e["sha256"]:
                raise ValueError("Shared-passage video no longer matches the frozen reviewed bytes")
            offset = e["source_interval"]["start_sec"]
            expected_duration = e["source_interval"]["end_sec"]-offset
            duration = a["sampling"]["source_duration_seconds"]
            fps = a["sampling"]["source_fps"]
            frame_count = a["sampling"].get("source_frame_count")
            if (any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
                    for v in (duration, fps, frame_count)) or int(frame_count) != frame_count):
                raise ValueError("Source duration, fps and frame count must be finite and positive")
            tolerance = max(.05, 2/fps)
            if abs(duration-expected_duration) > tolerance:
                raise ValueError("Cannot map source time: export duration does not match the recorded passage")
            bounds = {"start_seconds": max(0., start-offset), "end_seconds": min(end-offset, duration)}
            key = hashlib.sha256(json.dumps([e["sha256"], signature(), config, bounds], sort_keys=True).encode()).hexdigest()
            cache = Path(args.output).parent / "overlap-features" / (key+".json")
            last_frame_time = (frame_count-1)/fps
            if bounds["start_seconds"] <= 1e-9 and bounds["end_seconds"] > last_frame_time + 1e-9:
                segment = a
                reused = True
            elif cache.exists():
                segment = read(cache)
                verify_analysis(segment, e["sha256"], signature(), dict(config, **bounds))
                reused = False
            else:
                segment = analyze_video(str(path), **config, **bounds)
                segment.update(media_sha256=e["sha256"], extractor_sha256=signature(), extraction_config=dict(config, **bounds))
                write(cache, segment)
                reused = False
            stat_after = path.stat()
            if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
                raise ValueError("Video changed during shared-passage analysis")
            record[role] = {"example_id": e["id"], "sha256": e["sha256"], "export_bounds_seconds": bounds,
                            "features": numeric_features(segment), "quality": segment["quality"],
                            "sampling": segment["sampling"], "reused_full_export": reused,
                            "analysis_path": full[e["id"]]["analysis_path"] if reused else str(cache.resolve())}
        rows.append(record)
        write(args.output, {"schema_version": 1, "manifest_sha256": digest(args.manifest),
              "extractor_sha256": signature(), "complete": len(rows) == len(pairs),
              "purpose": "Shared-passage motion diagnostic only; subclips have no separate user labels and add no benchmark examples",
              "pairs": rows})
        print(json.dumps({"pair": pair["id"], "shared_seconds": round(end-start, 3), "complete": True}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2, help="OpenCV worker threads (default: 2)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="Verify frozen export hashes and extract video motion features")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-fps", type=float, default=1000., help="Sampling cap; default uses every native frame")
    p.add_argument("--width", type=int, default=640)
    p.set_defaults(run=extract)
    p = sub.add_parser("evaluate", help="Train fixed model; evaluate known examples and source-grouped holdout")
    p.add_argument("--manifest", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=evaluate)
    p = sub.add_parser("detect", help="Score a new export without changing its review status")
    p.add_argument("video")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model JSON (default: bundled native-v3 calibration)")
    p.add_argument("--output")
    p.set_defaults(run=detect)
    p = sub.add_parser("overlap", help="Compare motion over the shared source passage without creating new labels")
    p.add_argument("--manifest", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=overlap)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    import cv2
    cv2.setNumThreads(args.threads)
    try:
        args.run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
