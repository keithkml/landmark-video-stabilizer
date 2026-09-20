"""Small, smooth residual fields composed with an existing affine stabilization.

Coordinates in the residual model are native output pixels. Affine matrices map
reference to source in analysis-image coordinates (inverse sampling convention).
"""

import cv2
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_tracks(path):
    with np.load(path, allow_pickle=False) as data:
        points = data["reference_points"].astype(float)
        displacement = data["displacement"].astype(float)
        valid = data["valid"].astype(bool)
        # Backward-compatible name in the original measured woodpecker fixture.
        count = int(data["protected_count"] if "protected_count" in data else data["bark_count"])
    require(points.ndim == 2 and points.shape[1] == 2, "Expected P x 2 reference points")
    require(displacement.ndim == 3 and displacement.shape[1:] == points.shape,
            "Expected N x P x 2 displacements")
    require(len(displacement) >= 2 and valid.shape == displacement.shape[:2], "Invalid validity mask")
    require(0 <= count < len(points), "Invalid protected landmark count")
    require(np.isfinite(points).all() and np.isfinite(displacement[valid]).all(), "Nonfinite valid tracks")
    return points, displacement, valid, count


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def mask_weights(points, regions):
    """Union of soft regions; axis ramps within each region are multiplied.

For example x_fall=[250,340] is 1 left of 250, 0 right of 340, smoothly
interpolated in between. Use a zero-weight area to protect moving subjects.
"""
    points = np.asarray(points)
    result = np.zeros(points.shape[:-1], dtype=float)
    require(bool(regions), "At least one explicitly configured correction region is required")
    for region in regions:
        require(bool(region), "Empty correction region would deform the entire image")
        weight = np.ones_like(result)
        for name, bounds in region.items():
            require(name in {"x_fall", "y_fall", "x_rise", "y_rise"}, f"Unknown ramp: {name}")
            lo, hi = bounds
            require(np.isfinite([lo, hi]).all() and hi > lo, "Ramp bounds must increase")
            value = smoothstep((points[..., 0 if name[0] == "x" else 1] - lo) / (hi - lo))
            weight *= 1 - value if name.endswith("fall") else value
        result = np.maximum(result, weight)
    return result


def field_geometry(fields, size, limits):
    width, height = size
    require(fields.ndim == 4 and fields.shape[-1] == 2 and min(fields.shape[1:3]) >= 3,
            "Expected N x grid_height x grid_width x 2 fields")
    require(np.isfinite(fields).all(), "Nonfinite correction field")
    dx = np.gradient(fields[..., 0], (width - 1) / (fields.shape[2] - 1), axis=2)
    dy = np.gradient(fields[..., 0], (height - 1) / (fields.shape[1] - 1), axis=1)
    ex = np.gradient(fields[..., 1], (width - 1) / (fields.shape[2] - 1), axis=2)
    ey = np.gradient(fields[..., 1], (height - 1) / (fields.shape[1] - 1), axis=1)
    jacobian = (1 + dx) * (1 + ey) - dy * ex
    report = {
        "maximum_correction_native_px": float(np.linalg.norm(fields, axis=3).max()),
        "min_jacobian": float(jacobian.min()), "max_jacobian": float(jacobian.max()),
    }
    require(report["min_jacobian"] > limits["min_jacobian"], "Correction folds or compresses too far")
    require(report["max_jacobian"] < limits["max_jacobian"], "Correction expands too far")
    require(report["maximum_correction_native_px"] < limits["max_displacement_px"], "Correction exceeds bound")
    return report


def fit_fields(tracks_path, config):
    """Reproduce the original regularized Gaussian-basis residual correction.

Validation landmarks are excluded from fitting, but ARE used to choose sigma.
Their reported error is validation error, not an untouched final test result.
"""
    points, displacement, valid, protected_count = load_tracks(tracks_path)
    candidates = np.array([j for j in range(protected_count, len(points))
                           if valid[:, j].mean() > config["min_valid_fraction"]])
    stride = config["validation_stride"]
    require(stride >= 2, "validation_stride must be at least two")
    train = candidates[np.arange(len(candidates)) % stride != 0]
    validation = candidates[np.arange(len(candidates)) % stride == 0]
    require(len(train) >= 3 and len(validation) >= 1, "Too few well-tracked background landmarks")
    for j in candidates:
        good = np.flatnonzero(valid[:, j])
        for axis in range(2):
            displacement[:, j, axis] = np.interp(np.arange(len(displacement)), good, displacement[good, j, axis])
    centers = points[train]

    def basis(query, sigma):
        distances = ((query[:, None, :] - centers[None, :, :]) / sigma) ** 2
        return np.exp(-0.5 * distances.sum(2)) * mask_weights(query, config["regions"])[:, None]

    reports, selected = [], None
    require(config["ridge"] > 0, "Regularization must be positive")
    for sigma in config["sigmas"]:
        require(np.isfinite(sigma) and sigma > 0, "Sigma must be positive")
        design, held = basis(points[train], sigma), basis(points[validation], sigma)
        inverse = np.linalg.solve(design.T @ design + np.eye(len(centers)) * config["ridge"], design.T)
        coefficients = np.einsum("kp,tpc->tkc", inverse, displacement[:, train])
        predicted = np.einsum("pk,tkc->tpc", held, coefficients)
        residual = displacement[:, validation] - predicted
        pairs = valid[1:, validation] & valid[:-1, validation]
        require(pairs.any(), "No consecutive valid validation observations")
        steps = np.linalg.norm(np.diff(residual, axis=0), axis=2)[pairs]
        original = np.linalg.norm(np.diff(displacement[:, validation], axis=0), axis=2)[pairs]
        report = {
            "sigma": sigma, "held_out_points": len(validation),
            "original_step_max": float(original.max()), "new_step_max": float(steps.max()),
            "new_step_p99": float(np.percentile(steps, 99)),
            "reference_error_p99": float(np.percentile(np.linalg.norm(residual, axis=2)[valid[:, validation]], 99)),
        }
        reports.append(report)
        if selected is None or report["new_step_max"] < selected[0]["new_step_max"]:
            selected = report, coefficients
    require(selected is not None, "No candidate sigma configured")
    report, coefficients = selected
    width, height = config["output_size"]
    grid = config["grid_size"]
    require(width > 2 and height > 2 and grid >= 3, "Invalid output or grid dimensions")
    yy, xx = np.mgrid[:grid, :grid].astype(float)
    query = np.stack([xx.ravel() * (width - 1) / (grid - 1), yy.ravel() * (height - 1) / (grid - 1)], axis=1)
    fields = np.einsum("pk,tkc->tpc", basis(query, report["sigma"]), coefficients)
    fields = fields.reshape(len(displacement), grid, grid, 2).astype(np.float32)
    report = dict(report, **field_geometry(fields, (width, height), config["limits"]),
                  all_frames=len(displacement), fit_indices=train.tolist(), validation_indices=validation.tolist(),
                  validation_used_for_model_selection=True, continuous_motion_verified=False, listening_verified=False)
    return fields, report, reports


def native_matrix(matrix, analysis_width, source_width, crop):
    """Preserve pixel-center alignment when converting analysis to native coordinates."""
    require(np.shape(matrix) == (2, 3) and np.isfinite(matrix).all(), "Invalid affine matrix")
    require(analysis_width > 0 and source_width > 0, "Invalid analysis/native width")
    scale = source_width / analysis_width
    offset = (scale - 1) / 2
    result = np.array(matrix, dtype=float, copy=True)
    result[:, 2] = (scale * matrix[:, 2] + offset - matrix[:, :2] @ np.array([offset, offset])
                    + matrix[:, :2] @ np.asarray(crop[:2], dtype=float))
    require(np.linalg.det(result[:, :2]) > 0, "Affine is singular or reflects the image")
    return result


def sampling_map(matrix, field, size, source_size, support=2):
    """Compose both corrections, enforcing real source-pixel support at every output pixel."""
    width, height = size
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    if field is not None:
        require(np.isfinite(field).all(), "Nonfinite residual field")
        correction = cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)
        xx = xx + correction[..., 0]
        yy = yy + correction[..., 1]
    mx = (matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]).astype(np.float32)
    my = (matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]).astype(np.float32)
    require(np.isfinite(mx).all() and np.isfinite(my).all(), "Nonfinite sampling map")
    sw, sh = source_size
    require(mx.min() >= support and mx.max() <= sw - support - 1
            and my.min() >= support and my.max() <= sh - support - 1,
            "Crop requires pixels outside the source; reduce crop or correction")
    return mx, my
