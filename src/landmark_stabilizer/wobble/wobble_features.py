"""Read-only, pixel-based measurements of temporally varying image deformation.

These are motion *measurements*, not proof of a rolling-shutter camera defect.
Stabilizer rubberiness and atmospheric distortion can produce similar signals.
The preference classifier lives separately; this module never reads a label,
filename convention, content hash, or prior review decision to infer quality.

Coordinates and displacements are reported at a reference width of 640 pixels.
Pair displacements are normalized to a 1/12-second sampling interval. Features
therefore have consistent units when callers change extraction resolution/rate.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import math
import time
from typing import Any

import cv2
import numpy as np

from .storage import media_filename

EXTRACTOR_VERSION = "wobble-motion-v3"
REFERENCE_WIDTH = 640.0
REFERENCE_FPS = 12.0
GRID_COLUMNS = 8
GRID_ROWS = 6
FEATURE_NAMES = (
    "residual_p50_px", "residual_p90_px", "affine_deformation_px",
    "projective_deformation_px", "row_coherence_px", "local_coherence_px",
    "spatial_incoherence_px", "rigid_motion_px", "rotation_degrees",
    "scale_change", "track_count", "coverage", "row_coverage",
    "background_inlier_ratio", "forward_backward_error_px",
    "affine_fit_supported", "projective_fit_supported",
)


def _percentile(values: Any, q: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if array.size else 0.0


def _rms(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(vectors * vectors, axis=-1)))) if vectors.size else 0.0


def _grid_ids(points: np.ndarray, width: int, height: int) -> np.ndarray:
    columns = np.clip((points[:, 0] / width * GRID_COLUMNS).astype(int), 0, GRID_COLUMNS - 1)
    rows = np.clip((points[:, 1] / height * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
    return rows * GRID_COLUMNS + columns


def _gray(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape[1] > width:
        height = max(1, round(frame.shape[0] * width / frame.shape[1]))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _distributed_corners(gray: np.ndarray) -> np.ndarray:
    """Equal per-cell budgets prevent a textured bird from consuming all tracks."""
    height, width = gray.shape
    corners = []
    for row in range(GRID_ROWS):
        y0, y1 = row * height // GRID_ROWS, (row + 1) * height // GRID_ROWS
        for column in range(GRID_COLUMNS):
            x0, x1 = column * width // GRID_COLUMNS, (column + 1) * width // GRID_COLUMNS
            tile = gray[y0:y1, x0:x1]
            if min(tile.shape) < 5:
                continue
            found = cv2.goodFeaturesToTrack(
                tile, maxCorners=12, qualityLevel=0.025, minDistance=6, blockSize=5,
            )
            if found is not None:
                found[:, 0, :] += (x0, y0)
                corners.append(found)
    if not corners:
        return np.empty((0, 1, 2), np.float32)
    return np.concatenate(corners).astype(np.float32)


def _empty_pair(timestamp: float, interval: float) -> dict[str, Any]:
    return {
        "timestamp": timestamp, "valid": False, "_interval": interval,
        **{name: 0.0 for name in FEATURE_NAMES},
        "_row_vectors": np.zeros((GRID_ROWS, 2), dtype=float),
        "_deformation_vectors": np.zeros((GRID_ROWS * GRID_COLUMNS, 2), dtype=float),
        "_scale_vectors": np.zeros((1, 2), dtype=float),
        "_rotation_vectors": np.zeros((1, 2), dtype=float),
    }


def _distributed_model_support(origin: np.ndarray, mask: np.ndarray | None,
                               width: int, height: int) -> bool:
    """Do not extrapolate a model fitted to a concentrated moving foreground."""
    if mask is None:
        return False
    selected = origin[mask.ravel().astype(bool)]
    if len(selected) < 24 or len(selected) / len(origin) < 0.45:
        return False
    cells = _grid_ids(selected, width, height)
    occupied = np.bincount(cells, minlength=GRID_ROWS * GRID_COLUMNS).reshape(GRID_ROWS, GRID_COLUMNS) >= 3
    horizontal_span = float(np.ptp(selected[:, 0])) / width
    vertical_span = float(np.ptp(selected[:, 1])) / height
    return bool(occupied.sum() >= 12 and (occupied.sum(axis=1) >= 2).sum() >= 3
                and (occupied.sum(axis=0) >= 2).sum() >= 4
                and horizontal_span >= 0.6 and vertical_span >= 0.5)


def _measure_pair(previous: np.ndarray, current: np.ndarray, timestamp: float,
                  interval: float) -> dict[str, Any]:
    result = _empty_pair(timestamp, interval)
    height, width = previous.shape
    points = _distributed_corners(previous)
    if len(points) < 12:
        return result
    tracked, status, errors = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
    )
    if tracked is None:
        return result
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, tracked, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
    )
    if backward is None:
        return result
    fb_error = np.linalg.norm(points[:, 0] - backward[:, 0], axis=1)
    destination = tracked[:, 0]
    good = (
        (status[:, 0] > 0) & (back_status[:, 0] > 0) & (fb_error < 1.0)
        & (errors[:, 0] < 40) & np.isfinite(destination).all(axis=1)
        & (destination[:, 0] >= 2) & (destination[:, 0] < width - 2)
        & (destination[:, 1] >= 2) & (destination[:, 1] < height - 2)
    )
    origin, destination = points[good, 0], destination[good]
    result["track_count"] = int(len(origin))
    if len(origin) < 12:
        return result
    cells = _grid_ids(origin, width, height)
    counts = np.bincount(cells, minlength=GRID_ROWS * GRID_COLUMNS)
    occupied = counts >= 3
    result["coverage"] = float(np.mean(occupied))
    result["row_coverage"] = float(np.mean(occupied.reshape(GRID_ROWS, GRID_COLUMNS).sum(axis=1) >= 3))
    spatial_scale = REFERENCE_WIDTH / width
    motion_scale = spatial_scale / (REFERENCE_FPS * interval)
    result["forward_backward_error_px"] = _percentile(fb_error[good], 50) * spatial_scale
    # Do not require similarity inliers for support: true deformed rows are
    # precisely the tracks the similarity model cannot explain.
    result["valid"] = bool(len(origin) >= 36 and result["coverage"] >= 0.25
                           and result["row_coverage"] >= 0.5)
    cv2.setRNGSeed(0)
    similarity, similarity_mask = cv2.estimateAffinePartial2D(
        origin, destination, method=cv2.RANSAC, ransacReprojThreshold=2.0 / spatial_scale,
        maxIters=500, confidence=0.99, refineIters=10,
    )
    if similarity is None or not np.isfinite(similarity).all():
        result["valid"] = False
        return result
    result["background_inlier_ratio"] = float(similarity_mask.mean())
    predicted = cv2.transform(origin[None], similarity)[0]
    residual = (destination - predicted) * motion_scale
    residual_norm = np.linalg.norm(residual, axis=1)
    result["residual_p50_px"] = _percentile(residual_norm, 50)
    result["residual_p90_px"] = _percentile(residual_norm, 90)
    result["rigid_motion_px"] = _percentile(np.linalg.norm(predicted - origin, axis=1), 50) * motion_scale
    result["rotation_degrees"] = abs(math.degrees(math.atan2(similarity[1, 0], similarity[0, 0]))) / (REFERENCE_FPS * interval)
    result["scale_change"] = abs(float(np.hypot(similarity[0, 0], similarity[1, 0])) - 1.0) / (REFERENCE_FPS * interval)

    # Exclude gross independent motion, but permit coherent deformation of a
    # large part of the image. Cell medians add protection against foregrounds.
    clip = max(3.0, 4.0 * result["residual_p50_px"])
    background = residual_norm <= clip
    centers = np.array([
        ((column + 0.5) * width / GRID_COLUMNS, (row + 0.5) * height / GRID_ROWS)
        for row in range(GRID_ROWS) for column in range(GRID_COLUMNS)
    ], np.float32)
    similarity_grid = cv2.transform(centers[None], similarity)[0]
    radius = _rms((centers - (width / 2, height / 2)) * spatial_scale)
    # Signed motion is essential: absolute per-pair scale/rotation cannot
    # distinguish smooth zoom/pan from rapid alternating pumping. The radius
    # expresses these otherwise angular/unitless changes as image displacement.
    log_scale = math.log(max(float(np.hypot(similarity[0, 0], similarity[1, 0])), 1e-12))
    angle = math.atan2(similarity[1, 0], similarity[0, 0])
    result["_scale_vectors"] = np.array([[log_scale * radius / (REFERENCE_FPS * interval), 0.0]])
    result["_rotation_vectors"] = np.array([[angle * radius / (REFERENCE_FPS * interval), 0.0]])
    cell_vectors = np.full((GRID_ROWS * GRID_COLUMNS, 2), np.nan)
    for cell in np.flatnonzero(occupied):
        selected = (cells == cell) & background
        if selected.sum() >= 3:
            cell_vectors[cell] = np.median(residual[selected], axis=0)
    valid_cells = np.isfinite(cell_vectors).all(axis=1)
    row_vectors = np.full((GRID_ROWS, 2), np.nan)
    cell_field = cell_vectors.reshape(GRID_ROWS, GRID_COLUMNS, 2)
    for row in range(GRID_ROWS):
        row_valid = np.isfinite(cell_field[row]).all(axis=1)
        if row_valid.sum() >= 3:
            row_vectors[row] = np.median(cell_field[row, row_valid], axis=0)
    valid_rows = np.isfinite(row_vectors).all(axis=1)
    if valid_rows.sum() >= 3:
        # Remove any common residual translation left by robust global fitting.
        row_vectors[valid_rows] -= np.median(row_vectors[valid_rows], axis=0)
        result["row_coherence_px"] = _rms(row_vectors[valid_rows])
    smooth_vectors = []
    incoherent_vectors = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            if not np.isfinite(cell_field[row, column]).all():
                continue
            neighborhood = cell_field[max(0, row - 1):min(GRID_ROWS, row + 2),
                                      max(0, column - 1):min(GRID_COLUMNS, column + 2)].reshape(-1, 2)
            neighborhood = neighborhood[np.isfinite(neighborhood).all(axis=1)]
            if len(neighborhood) >= 4:
                smooth = np.median(neighborhood, axis=0)
                smooth_vectors.append(smooth)
                incoherent_vectors.append(cell_field[row, column] - smooth)
    result["local_coherence_px"] = _rms(np.asarray(smooth_vectors))
    result["spatial_incoherence_px"] = _rms(np.asarray(incoherent_vectors))

    cv2.setRNGSeed(0)
    affine, affine_mask = cv2.estimateAffine2D(
        origin, destination, method=cv2.RANSAC, ransacReprojThreshold=2.0 / spatial_scale,
        maxIters=500, confidence=0.99, refineIters=10,
    )
    deformation = np.zeros_like(centers, dtype=float)
    if affine is not None and np.isfinite(affine).all() and _distributed_model_support(origin, affine_mask, width, height):
        result["affine_fit_supported"] = 1.0
        deformation = (cv2.transform(centers[None], affine)[0] - similarity_grid) * motion_scale
        result["affine_deformation_px"] = _rms(deformation[occupied])
    cv2.setRNGSeed(0)
    homography, homography_mask = cv2.findHomography(
        origin, destination, cv2.RANSAC, 2.0 / spatial_scale,
        maxIters=500, confidence=0.99,
    )
    if homography is not None and np.isfinite(homography).all() and _distributed_model_support(origin, homography_mask, width, height):
        projective = (cv2.perspectiveTransform(centers[None], homography)[0] - similarity_grid) * motion_scale
        # Failed fits or distant poles should not turn into a quality alarm.
        if np.isfinite(projective).all() and _percentile(np.linalg.norm(projective[occupied], axis=1), 90) < max(30.0, 8.0 * result["residual_p90_px"]):
            result["projective_fit_supported"] = 1.0
            result["projective_deformation_px"] = _rms(projective[occupied])
            deformation = projective
    # Fill missing rows/cells with zero only for temporal measurements. Coverage
    # and valid-pair fractions remain explicit, so callers can abstain.
    result["_row_vectors"] = np.nan_to_num(row_vectors)
    result["_deformation_vectors"] = deformation
    return result


def _temporal_features(pairs: list[dict[str, Any]], key: str) -> tuple[float, float]:
    """Return brief direction reversals and their fraction of deformation energy.

    Compare within a quarter-second instead of only adjacent pairs. A stationary
    sample at a sinusoid's turning point must not hide the opposite directions
    immediately before and after it. The time horizon also avoids interpreting
    slow changes of perspective as rapid oscillation. Never bridge invalid data.
    """
    reversal, energy = [], []
    for index, current in enumerate(pairs):
        if not current["valid"]:
            continue
        strongest_reverse, corresponding_energy = 0.0, 0.0
        found_comparison = False
        for previous_index in range(index - 1, -1, -1):
            previous = pairs[previous_index]
            if current["timestamp"] - previous["timestamp"] > 0.25 + 1e-7:
                break
            if not previous["valid"]:
                break
            found_comparison = True
            left, right = previous[key], current[key]
            dot = np.sum(left * right, axis=-1)
            reverse = float(np.mean(np.maximum(0, -dot)))
            pair_energy = float(np.mean((np.sum(left * left, axis=-1)
                                         + np.sum(right * right, axis=-1)) / 2))
            if reverse >= strongest_reverse:
                strongest_reverse, corresponding_energy = reverse, pair_energy
        if found_comparison:
            reversal.append(strongest_reverse)
            energy.append(corresponding_energy)
    if not reversal:
        return 0.0, 0.0
    reverse_energy = float(np.mean(reversal))
    return math.sqrt(reverse_energy), reverse_energy / max(float(np.mean(energy)), 1e-12)


def _summarize(pairs: list[dict[str, Any]]) -> dict[str, float]:
    valid = [pair for pair in pairs if pair["valid"]]
    result: dict[str, float] = {"valid_pair_fraction": len(valid) / max(len(pairs), 1)}
    # Explicit fixed keys, even if no usable image evidence was recovered.
    for name in FEATURE_NAMES:
        values = [pair[name] for pair in valid]
        result[f"{name}_median"] = _percentile(values, 50)
        result[f"{name}_p90"] = _percentile(values, 90)
        result[f"{name}_p95"] = _percentile(values, 95)
    for label, key in (("deformation", "_deformation_vectors"), ("row", "_row_vectors")):
        amplitude, fraction = _temporal_features(pairs, key)
        result[f"{label}_oscillation_px"] = amplitude
        result[f"{label}_reversal_fraction"] = fraction
    for label, key in (("scale", "_scale_vectors"), ("rotation", "_rotation_vectors")):
        amplitude, fraction = _temporal_features(pairs, key)
        result[f"{label}_pumping_px"] = amplitude
        result[f"{label}_pumping_reversal_fraction"] = fraction
    return {name: round(float(value), 7) for name, value in result.items()}


def analyze_frames(frames: Iterable[tuple[float, np.ndarray]], *, width: int = 640,
                   window_seconds: float = 2.0, stride_seconds: float = 1.0,
                   progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Analyze timestamped BGR/gray frames, without retaining decoded video.

    Timestamps must increase. This public helper supports deterministic synthetic
    evals and callers that already have a streaming decoder.
    """
    if width < 128 or window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("width must be >=128; window and stride must be positive")
    previous, previous_time, first_time = None, None, None
    pairs: list[dict[str, Any]] = []
    sampled_frames, last_progress = 0, time.monotonic()
    shape = None
    for timestamp, frame in frames:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (previous_time is not None and timestamp <= previous_time):
            raise ValueError("frame timestamps must be finite and strictly increasing")
        current = _gray(frame, width)
        if shape is not None and current.shape != shape:
            raise ValueError("frame dimensions changed during analysis")
        shape = current.shape
        if first_time is None:
            first_time = timestamp
        if previous is not None:
            pairs.append(_measure_pair(previous, current, timestamp, timestamp - previous_time))
        previous, previous_time = current, timestamp
        sampled_frames += 1
        if progress_callback and time.monotonic() - last_progress >= 5:
            progress_callback({"sampled_frames": sampled_frames, "timestamp": timestamp})
            last_progress = time.monotonic()
    duration = 0.0 if first_time is None else float(previous_time - first_time)
    aggregate = _summarize(pairs)
    windows = []
    if pairs:
        start = float(first_time)
        while start < float(previous_time):
            end = min(start + window_seconds, float(previous_time))
            selected = [pair for pair in pairs if start < pair["timestamp"] <= end]
            if selected:
                windows.append({"start_seconds": round(start, 4), "end_seconds": round(end, 4),
                                "pair_count": len(selected), "features": _summarize(selected)})
            if end >= float(previous_time):
                break
            start += stride_seconds
    # Worst supported windows preserve brief defects that clip means conceal.
    for name in aggregate.copy():
        if name not in ("valid_pair_fraction",) and windows:
            supported = [window["features"][name] for window in windows
                         if window["features"]["valid_pair_fraction"] >= 0.5]
            aggregate[f"worst_window_{name}"] = _percentile(supported, 100)
        elif name != "valid_pair_fraction":
            aggregate[f"worst_window_{name}"] = 0.0
    reasons = []
    supported_duration = sum(pair["_interval"] for pair in pairs if pair["valid"])
    supported_fraction = supported_duration / duration if duration > 0 else 0.0
    unsupported_run, longest_unsupported = 0.0, 0.0
    for pair in pairs:
        unsupported_run = 0.0 if pair["valid"] else unsupported_run + pair["_interval"]
        longest_unsupported = max(longest_unsupported, unsupported_run)
    if sampled_frames < 13 or duration < 1.0:
        reasons.append("insufficient_duration")
    if supported_fraction < 0.85 - 1e-9:
        reasons.append("insufficient_texture_or_coverage")
    if longest_unsupported > 0.5 + 1e-9:
        reasons.append("unsupported_interval_exceeds_half_second")
    if aggregate["coverage_median"] < 0.35 or aggregate["row_coverage_median"] < 0.5:
        reasons.append("insufficient_spatial_coverage")
    return {
        "schema_version": 1, "extractor_version": EXTRACTOR_VERSION,
        "config": {"width": width, "window_seconds": window_seconds,
                   "stride_seconds": stride_seconds, "reference_width": REFERENCE_WIDTH,
                   "reference_fps": REFERENCE_FPS, "grid_columns": GRID_COLUMNS,
                   "grid_rows": GRID_ROWS},
        "sampling": {"sampled_frames": sampled_frames, "pair_count": len(pairs),
                     "analyzed_duration_seconds": round(duration, 6),
                     "analyzed_start_seconds": round(float(first_time), 6) if first_time is not None else None,
                     "analyzed_end_seconds": round(float(previous_time), 6) if previous_time is not None else None,
                     "analysis_width": shape[1] if shape else 0,
                     "analysis_height": shape[0] if shape else 0},
        "quality": {"usable": not reasons, "reasons": reasons,
                    "supported_duration_fraction": round(supported_fraction, 7),
                    "longest_unsupported_seconds": round(longest_unsupported, 7)},
        "aggregate": aggregate, "windows": windows,
    }


def analyze_video(path: str | Path, *, sample_fps: float = 1000.0, width: int = 640,
                  window_seconds: float = 2.0, stride_seconds: float = 1.0,
                  max_seconds: float | None = None,
                  start_seconds: float = 0.0, end_seconds: float | None = None,
                  progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Stream a video through the storage adapter and measure its image motion.

    No media or catalog mutation occurs. Decode/open failures raise instead of
    reporting zero wobble. Poor texture is an explicit quality/abstention signal.
    Segment timestamps stay relative to the original export. ``max_seconds``
    limits analyzed duration starting at ``start_seconds``; ``end_seconds`` is
    an exclusive export-relative bound. When both are given, the earlier wins.
    """
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and nonnegative")
    if end_seconds is not None and (not math.isfinite(end_seconds) or end_seconds <= start_seconds):
        raise ValueError("end_seconds must be finite and greater than start_seconds")
    stop_seconds = end_seconds
    if max_seconds is not None:
        stop_seconds = min(stop_seconds, start_seconds + max_seconds) if stop_seconds is not None else start_seconds + max_seconds
    cap = cv2.VideoCapture(str(media_filename(path)))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Unable to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Video does not report a valid frame rate: {path}")
    effective_fps = min(sample_fps, fps)
    source_duration = frame_count / fps
    decoded_frames = 0
    segment_decoded_frames = 0

    def frames() -> Iterable[tuple[float, np.ndarray]]:
        nonlocal decoded_frames, segment_decoded_frames
        frame_index, sample_index = 0, 0
        first_frame = math.ceil(start_seconds * fps - 1e-9)
        while True:
            timestamp = frame_index / fps
            if stop_seconds is not None and timestamp >= stop_seconds - 1e-9:
                break
            if not cap.grab():
                break
            decoded_frames += 1
            if frame_index >= first_frame:
                segment_decoded_frames += 1
            if frame_index >= first_frame + round(sample_index * fps / effective_fps):
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise ValueError(f"Failed to decode video frame {frame_index}: {path}")
                yield timestamp, frame
                sample_index += 1
            frame_index += 1

    try:
        result = analyze_frames(frames(), width=width, window_seconds=window_seconds,
                                stride_seconds=stride_seconds, progress_callback=progress_callback)
    finally:
        cap.release()
    if result["sampling"]["sampled_frames"] == 0:
        raise ValueError(f"Video contains no decodable frames: {path}")
    result["source"] = str(path)
    result["config"].update({"sample_fps": sample_fps, "max_seconds": max_seconds,
                             "start_seconds": start_seconds, "end_seconds": end_seconds})
    result["sampling"].update({"source_fps": fps, "effective_fps": effective_fps,
                                "source_frame_count": frame_count,
                                "source_duration_seconds": round(source_duration, 6),
                                "decoded_frames": decoded_frames,
                                "segment_decoded_frames": segment_decoded_frames})
    expected = min(frame_count, math.ceil(stop_seconds * fps - 1e-9)) if stop_seconds is not None else frame_count
    if frame_count > 0 and decoded_frames < max(1, expected - 2):
        result["quality"]["usable"] = False
        result["quality"]["reasons"].append("incomplete_decode")
    return result
