"""Source-grouped evaluation and a pixel-feature preference classifier.

No paths, species, source IDs, approval metadata or hashes enter the predictor.
Those fields are used exclusively to audit labels and isolate evaluation groups.
The fixed forest recipe is intentionally small enough to inspect and run without
an ML service. Scores are vote fractions, not calibrated probabilities.
"""
from __future__ import annotations

import math
from numbers import Real
from collections import Counter
from typing import Any

import numpy as np

LABELS = {"not_too_wobbly": 0, "too_wobbly": 1}
MODEL_VERSION = "wobble-preference-forest-v1"
RECIPE = {"trees": 128, "max_depth": 6, "min_leaf": 1, "thresholds": 7, "seed": 27092026}
# Physical motion measurements; the v3 revision adds scale/rotation pumping.
# Measurement support is a quality gate, not a predictor feature. Each run
# records its feature schema and implementation for development comparisons.
PREDICTION_FEATURES = (
    "residual_p50_px_p90", "residual_p90_px_p90", "affine_deformation_px_p90",
    "projective_deformation_px_p90", "row_coherence_px_p90", "local_coherence_px_p90",
    "spatial_incoherence_px_p90", "scale_change_p90", "rotation_degrees_p90",
    "deformation_oscillation_px", "row_oscillation_px", "deformation_reversal_fraction",
    "row_reversal_fraction", "worst_window_row_coherence_px_p90",
    "worst_window_local_coherence_px_p90", "worst_window_scale_change_p90",
    "worst_window_rotation_degrees_p90", "worst_window_deformation_oscillation_px",
    "worst_window_row_oscillation_px",
    "scale_pumping_px", "scale_pumping_reversal_fraction",
    "rotation_pumping_px", "rotation_pumping_reversal_fraction",
    "worst_window_scale_pumping_px", "worst_window_rotation_pumping_px",
)


def validate_examples(examples, pairs=()):
    ids, hashes, groups = {}, {}, {}
    for item in examples:
        if item["id"] in ids:
            raise ValueError(f"Duplicate example ID: {item['id']}")
        if item.get("label") not in LABELS or not item.get("source_id"):
            raise ValueError("Each example needs a valid user label and source_id")
        ids[item["id"]] = item
        group = item.get("split_group", item["source_id"])
        if not isinstance(group, str) or not group:
            raise ValueError("Every split group must be a nonempty string")
        if item["source_id"] in groups and groups[item["source_id"]] != group:
            raise ValueError("Versions of one source must share the same split group")
        groups[item["source_id"]] = group
        sha = item.get("sha256")
        if sha:
            if sha in hashes and hashes[sha] != (item["label"], item["source_id"]):
                raise ValueError("The same media hash has conflicting labels or source groups")
            hashes[sha] = (item["label"], item["source_id"])
    for pair in pairs:
        if pair.get("before_id") not in ids or pair.get("after_id") not in ids:
            raise ValueError("Pair references an unknown example")
        before, after = ids[pair["before_id"]], ids[pair["after_id"]]
        if before["source_id"] != after["source_id"]:
            raise ValueError("Before/after pair must belong to the same source")
        if before["label"] != "too_wobbly" or after["label"] != "not_too_wobbly":
            raise ValueError("Pair labels must be rejected before and accepted after")


def _vector(features, names):
    try:
        raw = [features[k] for k in names]
        if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) for v in raw):
            return None
        values = np.array(raw, dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return values if np.all(np.isfinite(values)) else None


def _weights(examples):
    # Equal class contribution, equal source contribution within each class,
    # and identical bytes never gain influence from duplicate registrations.
    counts = Counter((e["label"], e.get("split_group", e["source_id"])) for e in examples)
    groups = {label: {e.get("split_group", e["source_id"]) for e in examples if e["label"] == label} for label in LABELS}
    return np.array([1 / (len(groups[e["label"]]) * counts[e["label"], e.get("split_group", e["source_id"])])
                     for e in examples], dtype=float)


def _tree(x, y, weight, rng, depth=0):
    probability = float(np.average(y, weights=weight))
    leaf = {"value": probability, "n": len(y)}
    if depth >= RECIPE["max_depth"] or len(y) < 2 * RECIPE["min_leaf"] or probability in (0., 1.):
        return leaf
    best, best_loss = None, float("inf")
    for j in rng.permutation(x.shape[1])[:max(1, math.ceil(math.sqrt(x.shape[1])))]:
        values = np.unique(x[:, j])
        if len(values) < 2:
            continue
        candidates = (values[:-1] + values[1:]) / 2
        if len(candidates) > RECIPE["thresholds"]:
            candidates = rng.choice(candidates, RECIPE["thresholds"], replace=False)
        for threshold in candidates:
            left = x[:, j] <= threshold
            if min(left.sum(), (~left).sum()) < RECIPE["min_leaf"]:
                continue
            wl, wr = weight[left].sum(), weight[~left].sum()
            pl = float(np.dot(y[left], weight[left]) / wl)
            pr = float(np.dot(y[~left], weight[~left]) / wr)
            loss = wl * pl * (1-pl) + wr * pr * (1-pr)
            if loss < best_loss:
                best_loss, best = loss, (int(j), float(threshold), left)
    if best is None:
        return leaf
    j, threshold, left = best
    return {"feature": j, "threshold": threshold, "n": len(y),
            "left": _tree(x[left], y[left], weight[left], rng, depth+1),
            "right": _tree(x[~left], y[~left], weight[~left], rng, depth+1)}


def fit_model(examples: list[dict], feature_names=None) -> dict:
    validate_examples(examples)
    names = sorted(feature_names if feature_names is not None else
                   {k for e in examples for k, v in e.get("features", {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)})
    if not names:
        raise ValueError("No numeric motion features supplied")
    seen, usable = set(), []
    for e in examples:
        key = e.get("sha256") or e["id"]
        if key not in seen and e.get("features_available", True) is True and _vector(e.get("features", {}), names) is not None:
            usable.append(e)
            seen.add(key)
    if {e["label"] for e in usable} != set(LABELS):
        raise ValueError("Calibration needs usable examples from both user-label classes")
    x = np.vstack([_vector(e["features"], names) for e in usable])
    y = np.array([LABELS[e["label"]] for e in usable], dtype=float)
    weights, rng = _weights(usable), np.random.default_rng(RECIPE["seed"])
    trees = [_tree(x, y, weights, rng) for _ in range(RECIPE["trees"])]
    return {"schema_version": 1, "model_version": MODEL_VERSION, "recipe": dict(RECIPE),
            "feature_names": names, "threshold": .5, "trees": trees,
            "training_example_ids": [e["id"] for e in usable],
            "training_source_ids": sorted({e["source_id"] for e in usable}),
            "training_classes": dict(Counter(e["label"] for e in usable)),
            "score_meaning": "Forest vote fraction; not a calibrated probability",
            "purpose": "User tolerance for visible wobble; cannot establish physical rolling-shutter cause"}


def _vote(vector, tree):
    while "feature" in tree:
        tree = tree["left"] if vector[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tree["value"]


def validate_model(model):
    """Reject corrupt or incompatible models before they can return a pass."""
    if (not isinstance(model, dict) or model.get("schema_version") != 1
            or model.get("model_version") != MODEL_VERSION):
        raise ValueError("Unsupported wobble model schema/version")
    names, trees, threshold = model.get("feature_names"), model.get("trees"), model.get("threshold")
    if (not isinstance(names, list) or not names or not all(isinstance(k, str) for k in names)
            or len(set(names)) != len(names)):
        raise ValueError("Invalid model feature schema")
    if (not isinstance(threshold, Real) or isinstance(threshold, bool)
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("Invalid model decision threshold")
    if not isinstance(trees, list) or not trees:
        raise ValueError("Model has no decision trees")

    def visit(tree, depth=0):
        if not isinstance(tree, dict) or depth > 32:
            raise ValueError("Malformed model tree")
        if "feature" in tree:
            j, cut = tree["feature"], tree.get("threshold")
            if (not isinstance(j, int) or isinstance(j, bool) or not 0 <= j < len(names)
                    or not isinstance(cut, Real) or isinstance(cut, bool) or not math.isfinite(cut)):
                raise ValueError("Invalid model split")
            visit(tree.get("left"), depth+1)
            visit(tree.get("right"), depth+1)
        else:
            value = tree.get("value")
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Invalid model leaf score")
    for tree in trees:
        visit(tree)


def predict_features(features: dict, model: dict) -> dict:
    validate_model(model)
    vector = _vector(features, model["feature_names"])
    if vector is None:
        return {"prediction": "needs_review", "score": None, "reason": "Missing or non-finite required motion features"}
    score = float(np.mean([_vote(vector, tree) for tree in model["trees"]]))
    return {"prediction": "too_wobbly" if score >= model["threshold"] else "not_too_wobbly",
            "score": score, "margin": abs(score-model["threshold"]),
            "reason": "Pixel motion features compared with observed user preferences"}


def _prediction(example, model):
    result = predict_features(example.get("features", {}), model) if example.get("features_available", True) is True else {
        "prediction": "needs_review", "score": None, "reason": example.get("feature_error", "Insufficient measurable motion support")}
    return dict(result, id=example["id"], source_id=example["source_id"], expected=example["label"])


def _wilson(correct, total):
    if not total:
        return None
    z, p = 1.96, correct / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0., mid-half), min(1., mid+half)]


def summarize_predictions(predictions, pairs=()):
    confusion = dict(tp=0, tn=0, fp=0, fn=0, abstain=0)
    for row in predictions:
        if row["prediction"] == "needs_review":
            confusion["abstain"] += 1
        elif row["expected"] == "too_wobbly":
            confusion["tp" if row["prediction"] == "too_wobbly" else "fn"] += 1
        else:
            confusion["tn" if row["prediction"] == "not_too_wobbly" else "fp"] += 1
    total, correct = len(predictions), confusion["tp"]+confusion["tn"]
    lookup = {p["id"]: p for p in predictions}
    pair_rows = []
    for pair in pairs:
        b, a = lookup[pair["before_id"]], lookup[pair["after_id"]]
        pair_rows.append(dict(pair,
            correct=b["prediction"] == "too_wobbly" and a["prediction"] == "not_too_wobbly",
            before_prediction=b["prediction"], after_prediction=a["prediction"],
            score_decreased=None if b["score"] is None or a["score"] is None else a["score"] < b["score"]))
    source_accuracy = []
    for source in sorted({p["source_id"] for p in predictions}):
        rows = [p for p in predictions if p["source_id"] == source]
        source_accuracy.append(sum(p["prediction"] == p["expected"] for p in rows)/len(rows))
    positives = sum(p["expected"] == "too_wobbly" for p in predictions)
    negatives = total-positives
    return {"total": total, "correct": correct, "strict_accuracy": correct/total if total else None,
            "source_macro_accuracy": float(np.mean(source_accuracy)) if source_accuracy else None,
            "confusion": confusion, "too_wobbly_recall": confusion["tp"]/positives if positives else None,
            "accepted_specificity": confusion["tn"]/negatives if negatives else None,
            "descriptive_wilson_95": _wilson(correct, total),
            "interval_caveat": "Per-example descriptive interval; variants within a source are correlated",
            "pairs": {"total": len(pair_rows), "correct": sum(p["correct"] for p in pair_rows), "details": pair_rows},
            "predictions": predictions}


def evaluate_examples(examples: list[dict], pairs=(), feature_names=None) -> dict:
    validate_examples(examples, pairs)
    model = fit_model(examples, feature_names)
    names = model["feature_names"]
    fitted = [_prediction(e, model) for e in examples]
    heldout, folds = [], []
    for source in sorted({e.get("split_group", e["source_id"]) for e in examples}):
        train = [e for e in examples if e.get("split_group", e["source_id"]) != source]
        test = [e for e in examples if e.get("split_group", e["source_id"]) == source]
        train_sources = sorted({e["source_id"] for e in train})
        try:
            fold_model = fit_model(train, names)
            predictions = [_prediction(e, fold_model) for e in test]
        except ValueError as exc:
            predictions = [dict(id=e["id"], source_id=e["source_id"], expected=e["label"],
                                prediction="needs_review", score=None, reason=str(exc)) for e in test]
        for p in predictions:
            p["training_source_ids"] = train_sources
        heldout.extend(predictions)
        folds.append({"heldout_source_id": source, "heldout_split_group": source, "training_source_ids": train_sources,
                      "test_ids": [e["id"] for e in test]})
    return {"schema_version": 1, "model": model,
            "resubstitution": summarize_predictions(fitted, pairs),
            "group_holdout": dict(summarize_predictions(heldout, pairs), folds=folds,
                                  method="Leave one entire source recording out; fixed recipe; no threshold tuning on its labels"),
            "acceptance_gate": {"all_known_examples_correct": all(p["prediction"] == p["expected"] for p in fitted),
                                "all_heldout_examples_correct": all(p["prediction"] == p["expected"] for p in heldout),
                                "all_heldout_pairs_correct": bool(pairs) and all(p["correct"] for p in summarize_predictions(heldout, pairs)["pairs"]["details"]),
                                "automatic_approval_permitted": False}}
