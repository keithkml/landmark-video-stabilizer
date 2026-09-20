"""Measure actual encoded frames against caller-supplied stationary landmarks."""

import cv2
import numpy as np

from .core import require


def measure_export(video, reference_points, expected_frames=None, support=41):
    cap = cv2.VideoCapture(str(video))
    points = np.asarray(reference_points, dtype=np.float32).reshape(-1, 1, 2)
    require(len(points) > 0 and np.isfinite(points).all(), "No valid reference points")
    displacement, validity = [], []
    try:
        ok, frame = cap.read()
        require(ok, "Could not decode first frame")
        height, width = frame.shape[:2]
        reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        require((points[:, 0, 0] >= 0).all() and (points[:, 0, 0] < width).all()
                and (points[:, 0, 1] >= 0).all() and (points[:, 0, 1] < height).all(), "Landmarks outside export")
        seed = points.copy()
        while ok:
            require(frame.shape[:2] == (height, width), "Frame dimensions changed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current, status, error = cv2.calcOpticalFlowPyrLK(
                reference, gray, points, seed, winSize=(35, 35), maxLevel=3,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .002))
            require(current is not None, "Optical flow failed")
            back, backward, _ = cv2.calcOpticalFlowPyrLK(
                gray, reference, current, points.copy(), winSize=(35, 35), maxLevel=3,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .003))
            require(back is not None, "Backward optical flow failed")
            good = ((status[:, 0] > 0) & (backward[:, 0] > 0) & (error[:, 0] < 22)
                    & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < .7))
            seed = current.copy()
            displacement.append(current[:, 0] - points[:, 0])
            validity.append(good)
            ok, frame = cap.read()
    finally:
        cap.release()
    displacement, validity = np.asarray(displacement), np.asarray(validity)
    require(len(displacement) >= 2, "Need at least two decoded frames")
    require(expected_frames is None or len(displacement) == expected_frames, "Decoded frame count mismatch")
    positions = points[:, 0][None, ...] + displacement
    interior = ((positions[..., 0] > support) & (positions[..., 0] < width - support)
                & (positions[..., 1] > support) & (positions[..., 1] < height - support)).all(axis=0)
    steps = np.linalg.norm(np.diff(displacement, axis=0), axis=2)
    pair_valid = validity[:-1] & validity[1:]

    def summary(selection):
        if not selection.any():
            return None
        usable = pair_valid[:, selection]
        if not usable.any():
            return {"landmarks": int(selection.sum()), "no_valid_pairs": True}
        selected = steps[:, selection]
        worst = np.where(usable, selected, -np.inf)
        frame, point = np.unravel_index(worst.argmax(), worst.shape)
        return {"landmarks": int(selection.sum()), "maximum_step_px": float(worst.max()),
                "p99_step_px": float(np.percentile(selected[usable], 99)),
                "minimum_valid_fraction_per_frame": float(validity[:, selection].mean(axis=1).min()),
                "worst_frame_pair": [int(frame), int(frame + 1)],
                "worst_landmark_index": int(np.flatnonzero(selection)[point])}

    report = {"decoded_frames": len(displacement), "all_landmarks": summary(np.ones(len(points), bool)),
              "interior_landmarks": summary(interior), "edge_landmarks": summary(~interior),
              "support_margin_px": support, "continuous_motion_verified": False, "listening_verified": False,
              "quality_approved": False,
              "limitation": "Tracking confidence, frame inspection, continuous playback and listening require review."}
    return report, displacement, validity
