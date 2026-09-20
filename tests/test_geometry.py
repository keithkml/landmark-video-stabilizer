import numpy as np
import pytest

from landmark_stabilizer.core import field_geometry, mask_weights, native_matrix, sampling_map


def test_mask_protects_subject_and_blends_background():
    regions = [{"x_fall": [250, 340]}, {"x_fall": [420, 510], "y_fall": [140, 230]}]
    points = np.array([[200, 800], [600, 600], [295, 600], [465, 185]])
    np.testing.assert_allclose(mask_weights(points, regions), [1, 0, .5, .25])


def test_refuses_unbounded_mask():
    with pytest.raises(ValueError, match="Empty correction region"):
        mask_weights(np.zeros((2, 2)), [{}])


def test_pixel_centers_and_inverse_sampling_composition():
    affine = np.array([[1.02, -.03, 4], [.01, .98, -3]])
    native = native_matrix(affine, 540, 2160, [400, 500, 80, 60])
    output = np.array([20.25, 30.5])
    # Native center -> analysis center -> source analysis -> source native.
    expected = ((affine[:, :2] @ ((output + [400, 500] - 1.5) / 4) + affine[:, 2]) * 4 + 1.5)
    np.testing.assert_allclose(native[:, :2] @ output + native[:, 2], expected)


def test_out_of_bounds_crop_is_rejected():
    with pytest.raises(ValueError, match="outside the source"):
        sampling_map(np.array([[1., 0, 0], [0, 1, 0]]), None, (32, 32), (64, 64))


def test_one_composed_map_keeps_fixed_crop():
    matrix = np.array([[1., 0, 10], [0, 1, 12]])
    field = np.ones((3, 3, 2), np.float32)
    mx, my = sampling_map(matrix, field, (20, 20), (64, 64))
    assert mx[0, 0] == 11 and my[0, 0] == 13
    assert mx[-1, -1] == 30 and my[-1, -1] == 32


def test_folded_grid_is_rejected():
    fields = np.zeros((2, 5, 5, 2), np.float32)
    fields[..., 0] = -2 * np.linspace(0, 100, 5)
    with pytest.raises(ValueError, match="folds"):
        field_geometry(fields, (101, 101), {"min_jacobian": .8, "max_jacobian": 1.2, "max_displacement_px": 300})


def test_nonfinite_transform_and_field_are_rejected():
    with pytest.raises(ValueError, match="Invalid affine"):
        native_matrix(np.full((2, 3), np.nan), 100, 200, [20, 20, 30, 30])
    with pytest.raises(ValueError, match="Nonfinite"):
        sampling_map(np.array([[1., 0, 10], [0, 1, 10]]), np.full((3, 3, 2), np.nan), (20, 20), (64, 64))
