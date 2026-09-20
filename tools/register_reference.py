"""Re-run the original fixed-reference affine fit from source frames and LK seeds.

Masks, seed matrices and frame interval come from a recipe. Moving birds must be
outside the masks; this tool intentionally requires a manually designed recipe.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists")
    recipe = json.loads(args.recipe.read_text())
    settings = recipe["registration"]
    first, end = recipe["render"]["source_frames"]
    seeds = np.load(args.recipe.parent / settings["seeds"], allow_pickle=False)
    if seeds.shape != (end - first, 2, 3) or not np.isfinite(seeds).all():
        parser.error("Expected one finite 2x3 seed matrix per selected frame")
    cv2.setNumThreads(4)
    cap = cv2.VideoCapture(str(args.source))
    positions, validity, matrices, errors = [], [], [], []
    try:
        for _ in range(first):
            if not cap.grab():
                raise ValueError("Source ended before selected start")
        ok, image = cap.read()
        if not ok or image.shape[1::-1] != tuple(recipe["render"]["source_display_size"]):
            raise ValueError("Source display dimensions differ from the recipe")
        reference = cv2.cvtColor(cv2.resize(image, tuple(settings["analysis_size"])), cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(reference)
        for polygon in settings["mask_polygons"]:
            cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
        points = cv2.goodFeaturesToTrack(reference, 360, .007, 12, mask=mask, blockSize=7)
        if points is None:
            raise ValueError("No bark/landmark features detected")
        base = points[:, 0]
        hold = ((base[:, 0] // 45 + base[:, 1] // 45).astype(int) % 4 == 0)
        if hold.sum() < 30:
            raise ValueError("Too few reserved landmarks")
        for i, seed_matrix in enumerate(seeds):
            if i:
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"Source ended at selected frame {i}")
            gray = cv2.cvtColor(cv2.resize(image, tuple(settings["analysis_size"])), cv2.COLOR_BGR2GRAY)
            seed = (base @ seed_matrix[:, :2].T + seed_matrix[:, 2]).astype(np.float32).reshape(-1, 1, 2)
            current, status, error = cv2.calcOpticalFlowPyrLK(
                reference, gray, points, seed, winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, .003),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, reference, current, points.copy(), winSize=(31, 31), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .005),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            valid = ((status[:, 0] > 0) & (back_status[:, 0] > 0) & (error[:, 0] < 20)
                     & (np.linalg.norm(back[:, 0] - base, axis=1) < .7))
            a, b = base[valid & ~hold], current[:, 0][valid & ~hold]
            if len(a) <= 80:
                raise ValueError(f"Too few fit landmarks at frame {i}")
            matrix, inliers = cv2.estimateAffine2D(
                a, b, method=cv2.RANSAC, ransacReprojThreshold=1.1,
                maxIters=2000, confidence=.999, refineIters=15)
            if matrix is None or inliers.sum() <= 80 or (valid & hold).sum() <= 20:
                raise ValueError(f"Low-confidence registration at frame {i}")
            a, b = a[inliers[:, 0] > 0], b[inliers[:, 0] > 0]
            matrix = np.linalg.lstsq(np.c_[a, np.ones(len(a))], b, rcond=None)[0].T
            residual = current[:, 0] - (base @ matrix[:, :2].T + matrix[:, 2])
            errors.append(float(np.sqrt(np.mean(np.sum(residual[valid & hold] ** 2, axis=1)))))
            matrices.append(matrix)
            positions.append(current[:, 0])
            validity.append(valid)
            if i % 600 == 0:
                print(f"Tracked {i}/{len(seeds)}", flush=True)
    finally:
        cap.release()
    args.out.mkdir(parents=True)
    np.save(args.out / "affine.npy", matrices)
    np.savez_compressed(args.out / "correspondences.npz", reference=base, points=positions,
                        valid=validity, heldout=hold, affine=matrices)
    (args.out / "registration.json").write_text(json.dumps({
        "frames": len(matrices), "held_out_landmarks": int(hold.sum()),
        "maximum_holdout_rms_analysis_px": max(errors), "continuous_motion_verified": False,
        "limitation": "Registration residual is not actual-export motion or perceptual review."
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
