"""Physics-oriented controls for motion extraction, independent of review labels."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from landmark_stabilizer.wobble.wobble_features import analyze_frames, analyze_video


def _texture(width=320, height=240):
    rng = np.random.default_rng(217)
    image = rng.integers(0, 256, (height, width), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.7)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def _frames(kind, count=48):
    background = _texture()
    height, width = background.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    for i in range(count):
        phase = i * 2 * np.pi * 2.0 / 12.0
        if kind == "stationary":
            frame = background.copy()
        elif kind == "low_texture":
            frame = np.full_like(background, 127)
        elif kind in ("rigid", "pan", "gradual_zoom", "pumping_zoom", "gradual_rotation"):
            angle = 1.5 * np.sin(phase) if kind == "rigid" else (i * 0.08 if kind == "gradual_rotation" else 0)
            scale = 1 + (i * 0.002 if kind == "gradual_zoom" else 0.035 * np.sin(phase) if kind == "pumping_zoom" else 0)
            affine = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            if kind in ("rigid", "pan"):
                affine[:, 2] += ((5 * np.sin(phase), 3 * np.cos(phase)) if kind == "rigid" else (i * 0.6, 0))
            frame = cv2.warpAffine(background, affine, (width, height), borderMode=cv2.BORDER_REFLECT101)
        elif kind == "row_wobble":
            displacement = 5 * np.sin(phase) * (0.5 * np.sin(2 * np.pi * yy / height) + (yy / height - 0.5))
            frame = cv2.remap(background, (xx + displacement).astype(np.float32), yy,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        elif kind == "foreground":
            frame = background.copy()
            x, y = int(110 + 24 * np.sin(phase / 4)), int(90 + 10 * np.cos(phase / 4))
            frame[y:y + 48, x:x + 64] = np.flipud(background[:48, :64])
        else:
            raise AssertionError(kind)
        yield i / 12.0, frame


@pytest.fixture(scope="module")
def measured():
    return {kind: analyze_frames(_frames(kind), width=320)
            for kind in ("stationary", "low_texture", "rigid", "pan", "row_wobble", "foreground",
                         "gradual_zoom", "pumping_zoom", "gradual_rotation")}


def test_stationary_and_low_texture_are_distinct(measured):
    stationary, flat = measured["stationary"], measured["low_texture"]
    assert stationary["quality"]["usable"]
    assert stationary["aggregate"]["residual_p90_px_p90"] < 0.02
    assert not flat["quality"]["usable"]
    assert "insufficient_texture_or_coverage" in flat["quality"]["reasons"]
    assert flat["aggregate"]["valid_pair_fraction"] == 0


def test_rigid_motion_and_smooth_pan_do_not_look_like_row_wobble(measured):
    wobble = measured["row_wobble"]["aggregate"]
    for kind in ("rigid", "pan"):
        control = measured[kind]
        assert control["quality"]["usable"]
        assert control["aggregate"]["rigid_motion_px_median"] > 0.5
        assert control["aggregate"]["row_coherence_px_p90"] < wobble["row_coherence_px_p90"] / 3
        assert control["aggregate"]["deformation_oscillation_px"] < wobble["deformation_oscillation_px"] / 3


def test_row_wobble_is_spatially_supported_and_oscillatory(measured):
    result = measured["row_wobble"]
    assert result["quality"]["usable"]
    features = result["aggregate"]
    assert features["coverage_median"] > 0.85
    assert features["row_coherence_px_p90"] > 0.4
    assert features["row_oscillation_px"] > 0.1
    assert features["projective_deformation_px_p90"] > 0.5


def test_small_moving_foreground_does_not_dominate_deformation(measured):
    foreground, wobble = measured["foreground"], measured["row_wobble"]
    assert foreground["quality"]["usable"]
    for key in ("row_coherence_px_p90", "projective_deformation_px_p90", "deformation_oscillation_px"):
        assert foreground["aggregate"][key] < wobble["aggregate"][key] / 3


def test_stable_schema_json_and_timestamped_windows(measured):
    reference = set(measured["stationary"]["aggregate"])
    for result in measured.values():
        assert set(result["aggregate"]) == reference
        json.dumps(result, allow_nan=False)
        assert result["sampling"]["sampled_frames"] == 48
        assert result["sampling"]["pair_count"] == 47
        assert result["windows"][0]["start_seconds"] == 0
        assert result["windows"][0]["end_seconds"] == 2


def test_video_decoding_is_read_only_and_samples_source(tmp_path):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (320, 240))
    assert writer.isOpened()
    for _, frame in _frames("row_wobble", count=48):
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    before = path.read_bytes()
    result = analyze_video(path, sample_fps=12, width=320)
    assert result["sampling"]["decoded_frames"] == 48
    assert result["sampling"]["sampled_frames"] == 24
    assert result["sampling"]["source_duration_seconds"] == 2
    assert result["quality"]["usable"]
    assert path.read_bytes() == before


def test_rejects_missing_video_and_invalid_timestamps(tmp_path):
    with pytest.raises(ValueError, match="Unable to open video"):
        analyze_video(tmp_path / "missing.mp4")
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_frames([(0, _texture()), (0, _texture())])
    with pytest.raises(ValueError, match="sample_fps"):
        analyze_video(Path("unused"), sample_fps=0)


def test_identical_pixels_ignore_source_filename(tmp_path):
    path = tmp_path / "rejected_before.avi"
    renamed = tmp_path / "approved_after.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (320, 240))
    for _, frame in _frames("stationary", count=18):
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    renamed.write_bytes(path.read_bytes())
    first, second = analyze_video(path, width=320), analyze_video(renamed, width=320)
    assert first["aggregate"] == second["aggregate"]
    assert first["quality"] == second["quality"]


def test_segment_bounds_preserve_export_timestamps(tmp_path):
    path = tmp_path / "segment.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (320, 240))
    assert writer.isOpened()
    for _, frame in _frames("pan", count=48):
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    result = analyze_video(path, width=320, start_seconds=1, end_seconds=3, max_seconds=1.5)
    assert result["sampling"]["analyzed_start_seconds"] == 1
    assert result["sampling"]["analyzed_end_seconds"] == pytest.approx(29 / 12, abs=1e-6)
    assert result["sampling"]["decoded_frames"] == 30
    assert result["sampling"]["segment_decoded_frames"] == 18
    assert result["sampling"]["sampled_frames"] == 18
    assert result["windows"][0]["start_seconds"] == 1
    assert result["quality"]["usable"]
    for bounds in ({"start_seconds": -1}, {"start_seconds": float("nan")},
                   {"start_seconds": 2, "end_seconds": 2}, {"end_seconds": float("inf")}):
        with pytest.raises(ValueError):
            analyze_video(path, **bounds)
    with pytest.raises(ValueError, match="no decodable frames"):
        analyze_video(path, start_seconds=10)


def test_signed_pumping_separates_smooth_zoom_and_rotation(measured):
    gradual, pumping = measured["gradual_zoom"], measured["pumping_zoom"]
    assert gradual["quality"]["usable"] and pumping["quality"]["usable"]
    assert gradual["aggregate"]["scale_change_median"] > 0.001
    assert gradual["aggregate"]["scale_pumping_px"] < 0.01
    assert pumping["aggregate"]["scale_pumping_px"] > 1.0
    assert pumping["aggregate"]["row_coherence_px_p90"] < 0.1
    assert measured["gradual_rotation"]["aggregate"]["rotation_pumping_px"] < 0.01
    assert measured["rigid"]["aggregate"]["rotation_pumping_px"] > 0.5
    # Rigid shake is measured separately; it must not inflate deformation.
    assert measured["rigid"]["aggregate"]["deformation_oscillation_px"] < 0.1


def test_partial_untrackable_clip_does_not_certify_acceptance():
    texture, flat = _texture(), np.full((240, 320), 127, np.uint8)
    result = analyze_frames(((i / 12, texture if i < 36 else flat) for i in range(60)), width=320)
    assert not result["quality"]["usable"]
    assert result["quality"]["supported_duration_fraction"] < 0.85
    assert result["quality"]["longest_unsupported_seconds"] > 0.5
    assert any(window["features"]["valid_pair_fraction"] == 1 for window in result["windows"])


def test_single_long_unsupported_gap_is_not_hidden_by_clip_length():
    texture, flat = _texture(), np.full((240, 320), 127, np.uint8)
    result = analyze_frames(((i / 12, flat if 40 <= i < 46 else texture) for i in range(120)), width=320)
    assert result["quality"]["supported_duration_fraction"] >= 0.85
    assert not result["quality"]["usable"]
    assert "unsupported_interval_exceeds_half_second" in result["quality"]["reasons"]


def test_default_native_sampling_avoids_a_fixed_twelve_fps_alias(tmp_path):
    path = tmp_path / "rapid_shear.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (320, 240))
    assert writer.isOpened()
    texture = _texture()
    yy, xx = np.mgrid[:240, :320].astype(np.float32)
    for index in range(72):
        displacement = 5 * np.sin(2 * np.pi * 6 * index / 24) * (yy / 240 - 0.5)
        frame = cv2.remap(texture, (xx + displacement).astype(np.float32), yy,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    native, undersampled = analyze_video(path, width=320), analyze_video(path, width=320, sample_fps=12)
    assert native["sampling"]["sampled_frames"] == 72
    assert native["sampling"]["effective_fps"] == 24
    assert native["aggregate"]["deformation_oscillation_px"] > 1
    assert undersampled["aggregate"]["deformation_oscillation_px"] < 0.01


def test_concentrated_foreground_cannot_support_extrapolated_model():
    from landmark_stabilizer.wobble.wobble_features import _distributed_model_support

    xx, yy = np.meshgrid(np.linspace(5, 315, 32), np.linspace(5, 235, 24))
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    full = np.ones((len(points), 1), np.uint8)
    assert _distributed_model_support(points, full, 320, 240)
    # Many points in one half are still not evidence for a full-image warp.
    concentrated = (points[:, 0] < 166).astype(np.uint8).reshape(-1, 1)
    assert concentrated.mean() > 0.45
    assert not _distributed_model_support(points, concentrated, 320, 240)
