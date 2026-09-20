import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from landmark_stabilizer.measure import measure_export
from landmark_stabilizer.render import render


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg integration test")
def test_actual_encoded_result_and_overwrite_protection(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x160:rate=30",
                    "-frames:v", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
    plan = tmp_path / "plan"
    plan.mkdir()
    np.save(plan / "affine.npy", np.tile([[1., 0, 0], [0, 1, 0]], (8, 1, 1)))
    np.save(plan / "residual-fields.npy", np.zeros((8, 3, 3, 2), np.float32))
    recipe = {"residual": {"output_size": [128, 128]},
              "render": {"source_frames": [0, 8], "crop": [16, 16, 128, 128],
                         "source_display_size": [160, 160], "analysis_width": 160, "fps": "30"}}
    output = tmp_path / "output.mp4"
    result = render(recipe, plan, source, output)
    assert result["full_decode_passed"] and result["frames"] == 8
    assert not result["quality_approved"] and result["audio"] == "video only"
    metadata = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)]))
    video = metadata["streams"][0]
    assert (video["width"], video["height"], int(video["nb_frames"])) == (128, 128, 8)
    assert video["color_space"] == "bt709" and video["color_range"] == "tv"
    before = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        render(recipe, plan, source, output)
    assert output.read_bytes() == before
    cap = cv2.VideoCapture(str(output))
    ok, image = cap.read()
    cap.release()
    assert ok
    points = cv2.goodFeaturesToTrack(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 20, .01, 10)[:, 0]
    report, displacement, valid = measure_export(output, points, expected_frames=8)
    assert report["decoded_frames"] == 8
    assert not report["quality_approved"] and not report["continuous_motion_verified"]
    assert displacement.shape == (8, len(points), 2) and valid.shape == (8, len(points))
    with pytest.raises(ValueError, match="frame count mismatch"):
        measure_export(output, points, expected_frames=9)
