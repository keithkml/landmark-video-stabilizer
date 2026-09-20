import json
from pathlib import Path

import numpy as np
import pytest

from landmark_stabilizer.core import fit_fields, mask_weights, native_matrix, sampling_map


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/woodpecker"


@pytest.fixture(scope="module")
def fitted():
    recipe = json.loads((EXAMPLE / "recipe.json").read_text())
    return recipe, fit_fields(EXAMPLE / recipe["tracks"], recipe["residual"])


def test_complete_recording_matches_successful_original_fit(fitted):
    recipe, (fields, report, comparisons) = fitted
    expected = json.loads((EXAMPLE / "expected-fit.json").read_text())
    assert len(fields) == expected["all_frames"] == 2910
    assert report["sigma"] == expected["sigma"] == 320
    assert report["held_out_points"] == expected["held_out_points"] == 15
    for key in ("original_step_max", "new_step_max", "new_step_p99", "reference_error_p99",
                "maximum_correction_native_px", "min_jacobian", "max_jacobian"):
        assert report[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-5), key
    assert len(comparisons) == 4
    assert set(report["fit_indices"]).isdisjoint(report["validation_indices"])
    assert report["validation_used_for_model_selection"]
    assert not report["continuous_motion_verified"]
    with np.load(EXAMPLE / "expected-fields.npz", allow_pickle=False) as baseline:
        np.testing.assert_allclose(fields[baseline["frames"]], baseline["fields"], rtol=2e-5, atol=2e-5)


def test_protected_area_has_no_extra_background_warp(fitted):
    recipe, (fields, _, _) = fitted
    yy, xx = np.mgrid[:65, :65].astype(float)
    points = np.stack([xx * 1239 / 64, yy * 1239 / 64], axis=-1)
    protected = mask_weights(points, recipe["residual"]["regions"]) == 0
    assert protected.any()
    assert np.max(np.abs(fields[:, protected])) == 0


def test_composed_native_maps_match_original_geometry(fitted):
    recipe, (fields, _, _) = fitted
    affine = np.load(EXAMPLE / "affine.npy", allow_pickle=False)
    for frame in (0, 1913, 2789, 2909):
        matrix = native_matrix(affine[frame], 1080, 2160, recipe["render"]["crop"])
        mx, my = sampling_map(matrix, fields[frame], (1240, 1240), (2160, 3840))
        assert mx.shape == my.shape == (1240, 1240)
