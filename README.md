# Landmark Video Stabilizer

Custom stabilization for difficult wildlife footage, with separate corrections
for stationary foreground landmarks and residual background movement. Extracted
from the successful September 2026 Red-headed Woodpecker repair.

The toolkit combines a fixed-reference affine correction with a small, smooth
background displacement field, then samples the original camera frame once.
The correction regions are chosen explicitly so that the bird keeps its natural
movement. It needs a shot-specific recipe and reliable measured landmarks; it is
not an automatic bird detector or a universal stabilization preset.

## Included

- Regularized Gaussian-basis fitting of residual landmark movement.
- Soft correction regions with an explicitly protected subject area.
- Bounds on deformation and correction magnitude.
- RGB16 rendering through FFmpeg and OpenCV, with one spatial resampling pass.
- Per-frame measurements on the actual encoded export.
- The successful woodpecker recipe, measured input tracks, affine matrices,
  original registration seeds, regression fixtures, and an honest evidence record.
- The original fixed-reference trunk registration procedure, made configurable.

Camera recordings, finished videos, proprietary LUTs, audio, and review-app data
are kept outside this repository. The example's numerical fixtures total about
2.7 MB. Generated plans and media go in ignored `work/` and `media/` directories.

## Install and reproduce the correction

Python 3.11 or newer is required. Install FFmpeg with `ffmpeg` and `ffprobe` on
PATH for rendering and video checks. The fit itself needs no camera media.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
landmark-stabilizer fit examples/woodpecker/recipe.json --out work/woodpecker
```

This fits all 2,910 frames and writes a roughly 94 MiB residual-field array,
affine matrices, the recipe snapshot, and numerical reports. The tests compare
the selected model, full-recording statistics, and sampled displacement grids
against the saved successful correction.

## Render from the original recording

The example requires the original `C0120.MP4` and its Sony
`1_SGamut3CineSLog3_To_LC-709.cube` LUT. The recipe retains the original two tone
curves and 1240 x 1240 native crop. Supply local paths when running it:

```sh
landmark-stabilizer render examples/woodpecker/recipe.json \
  --plan work/woodpecker \
  --source /path/to/C0120.MP4 \
  --lut /path/to/1_SGamut3CineSLog3_To_LC-709.cube \
  --output work/woodpecker/new-render.mp4
```

This creates **video only**. To retain sound, add `--audio /path/to/aligned.m4a`
with an AAC track already cut and aligned to the selected source interval. The
audio stream is copied; this package does not denoise, normalize, or retime it.
The original example used an independently prepared field-audio track measured
at -20.38 dBTP. Listen to any new export and verify its levels and synchronization.

Use `--limit-frames 12` and a different output name for a short smoke render.
Existing exports, plan directories, and logs are never intentionally overwritten.
Each render records its commands and SHA-256. Full video decoding is checked
before the final output filename is created.

Reproducing the **geometry** is the regression target. Byte-identical MP4 output
also depends on the source, LUT, prepared audio, encoder build, and libraries.

## Measure the actual export

```sh
landmark-stabilizer measure work/woodpecker/new-render.mp4 \
  --points examples/woodpecker/export-landmarks.npz \
  --expected-frames 2910 \
  --out work/woodpecker/check
```

The command checks fixed-reference landmarks on every decoded frame, reports
tracking confidence and the worst adjacent-frame displacement, and saves the
raw measurements. Interior and edge landmarks are reported separately using a
fixed support margin; large residuals are not silently discarded.

Measurements cannot establish perceptual quality. Inspect framing and individual
failure frames, watch the entire export in motion, and listen to its audio. The
tools leave `quality_approved`, `continuous_motion_verified`, and
`listening_verified` false. They do not approve or publish media.

## Re-run the foreground registration

The included recipe also preserves the original bark masks and coarse tracking
seeds. This reruns the first-stage affine fit from the private camera file:

```sh
python tools/register_reference.py examples/woodpecker/recipe.json \
  --source /path/to/C0120.MP4 --out work/woodpecker-registration
```

This operation is optional for reproducing the residual correction: the exact
successful affine matrices are already included. It can take several minutes.
OpenCV feature detection and robust fitting can differ across library versions;
inspect a regenerated registration before replacing the saved matrices.

## Adapting another recording

The reusable core is in `src/landmark_stabilizer/`. The example contains the
decisions specific to this shot: stable tracking areas, camera geometry, crop,
frame range, grade, and protected subject region.

For another shot, first establish a reliable foreground registration and measure
the remaining motion in its baseline export. Supply the arrays described in
[the coordinate and data guide](docs/algorithm.md), choose correction regions
that exclude the bird throughout the sequence, then fit and render a new plan.
Fast-moving birds, occlusion, focus loss, and independently moving leaves require
different tracking choices. Do not treat moving foliage as a fixed camera anchor.

The example records the final result and its limits in
[export-evidence.json](examples/woodpecker/export-evidence.json). The provenance
file records hashes of the original working scripts and the committed fixtures.
