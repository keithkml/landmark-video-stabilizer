import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .core import fit_fields, require
from .measure import measure_export
from .render import render


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Landmark-guided regional video stabilization")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="Fit the residual correction from saved landmark measurements")
    fit.add_argument("recipe", type=Path)
    fit.add_argument("--out", required=True, type=Path)
    rendering = sub.add_parser("render", help="Render original source pixels using a fitted plan")
    rendering.add_argument("recipe", type=Path)
    rendering.add_argument("--plan", required=True, type=Path)
    rendering.add_argument("--source", required=True, type=Path)
    rendering.add_argument("--output", required=True, type=Path)
    rendering.add_argument("--lut", type=Path)
    rendering.add_argument("--audio", type=Path, help="Optional AAC audio already aligned to the selected passage")
    rendering.add_argument("--limit-frames", type=int, help="Short smoke render; default is the complete selected interval")
    rendering.add_argument("--ffmpeg", default="ffmpeg")
    measurement = sub.add_parser("measure", help="Measure landmarks on every actual exported frame")
    measurement.add_argument("video", type=Path)
    measurement.add_argument("--points", required=True, type=Path, help="NPZ containing reference_points in output pixels")
    measurement.add_argument("--expected-frames", type=int)
    measurement.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    cv2.setNumThreads(3)
    try:
        if args.command == "fit":
            recipe = json.loads(args.recipe.read_text())
            require(not args.out.exists(), "Plan directory exists; choose a new one")
            fields, report, comparison = fit_fields(args.recipe.parent / recipe["tracks"], recipe["residual"])
            affine = np.load(args.recipe.parent / recipe["affine"], allow_pickle=False)
            require(affine.shape == (len(fields), 2, 3) and np.isfinite(affine).all(), "Affine shape mismatch")
            args.out.mkdir(parents=True)
            np.save(args.out / "residual-fields.npy", fields)
            np.save(args.out / "affine.npy", affine)
            write_json(args.out / "fit-report.json", report)
            write_json(args.out / "model-comparison.json", comparison)
            write_json(args.out / "recipe.json", recipe)
            print(json.dumps(report, indent=2))
        elif args.command == "render":
            recipe = json.loads(args.recipe.read_text())
            plan_recipe = json.loads((args.plan / "recipe.json").read_text())
            require(recipe == plan_recipe, "Recipe changed since fitting; create a new plan")
            report = render(recipe, args.plan, args.source, args.output, args.lut, args.audio,
                            args.limit_frames, args.ffmpeg)
            print(json.dumps(report, indent=2))
        else:
            require(not args.out.exists(), "Measurement directory exists; choose a new one")
            with np.load(args.points, allow_pickle=False) as data:
                points = data["reference_points"]
            report, displacement, valid = measure_export(args.video, points, args.expected_frames)
            args.out.mkdir(parents=True)
            np.savez_compressed(args.out / "motion.npz", reference_points=points,
                                displacement=displacement, valid=valid)
            write_json(args.out / "measurements.json", report)
            print(json.dumps(report, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"error: {error}\n")
