# Advisory wobble detector

The detector measures distributed image motion in a finished video and compares
it with a frozen calibration of user-rejected and user-approved exports. It is
useful for finding possible deformation, row-dependent movement, and repeated
scale/rotation pumping. It does not stabilize a video, diagnose a physical
rolling-shutter cause, or approve an export.

## Score a video

The normal package installation includes the detector and its small native-v3
model. No API key, network service, catalog, or separate model download is needed.

```sh
landmark-stabilizer wobble /path/to/export.mp4 --output work/wobble.json
# Equivalent standalone command:
wobble-detector detect /path/to/export.mp4 --output work/another-result.json
```

Use `--model /path/to/model.json` to select another compatible calibration.
The main command accepts `--threads 2`; for the standalone tool use
`wobble-detector --threads 2 detect ...`.

The terminal shows `prediction`, `score`, media/model SHA-256, and validation
context. The JSON file also contains tracking quality, aggregate measurements,
and two-second windows with one-second strides. The default samples every native
frame, up to a 1000 fps cap, at a maximum analysis width of 640 pixels. High-rate
sampling matters: sparse frames can miss rapid wobble.

- `too_wobbly`: the feature vector falls on the rejected side of the classifier.
- `not_too_wobbly`: the feature vector falls on the accepted side; this is not
  proof of a good video or a user approval.
- `needs_review`: insufficient measurable motion support, including low texture
  or incomplete coverage. A null score is not zero wobble.

The score is a forest vote fraction, **not calibrated severity or probability**.
Unusable/unreadable files and incompatible code/model hashes are errors, not
passes. All three predictions exit successfully; command/input errors exit 2.
Every result includes `automatic_approval_permitted: false`.

Detection reads media without modifying it and refuses to overwrite an existing
output, including the input video, model, symlink, or a previous result. It reads
the exact local path supplied; it does not redirect paths to a media drive.
Without `--output`, only the compact terminal summary is emitted.

## What the calibration establishes

The frozen corpus contains **nine exact exports**, five comparisons, and only
**three independent recordings**. There are five rejected and four later
user-approved examples. The fitted model reproduces all nine labels. Development
evaluation holding out every variant of one recording at a time scores **6/9**,
with three false alarms on accepted exports. The corpus informed development;
these are not results on an untouched prospective test set.

Some comparisons use overlapping recuts rather than identical passages. Those
distinctions are preserved in the regression fixture. Natural foliage motion,
moving subjects, parallax, changing framing, and atmospheric distortion can
resemble or change the measured deformation. Smooth translation alone is not
the same problem as wobble: this is not a comprehensive camera-jitter detector.
Watch the actual exported video and listen separately before approving it.

## Port provenance and reproducibility

`src/landmark_stabilizer/wobble/wobble.py`, `wobble_features.py`, and
`data/native-v3.json` preserve the original calibrated bytes. The CLI verifies
the predictor and extractor hashes stored in the model. `storage.py` is a small
local-path adapter replacing the original workspace's volume routing; it does
not change image measurements. The CLI was adapted for package imports and
exclusive result-file creation. `data/provenance.json` records this boundary.

`tests/fixtures/wobble-native-v3.json` contains the numeric features, opaque
export/source identifiers, media hashes, pair definitions, and expected scores.
It contains no media, private paths, species, coordinates, or user messages.
Tests reproduce the saved trees, exact scores, and source-holdout evaluation;
synthetic encoded videos exercise full decoding, abstention, and input safety.
These regression checks do not amount to re-extracting the private nine videos
on every platform. OpenCV/decoder versions can affect measurements on new media.

## Recalibrate with your own reviewed examples

The standalone CLI also retains `extract`, `evaluate`, and `overlap`:

```sh
wobble-detector extract --manifest corpus.json --output work/extraction
wobble-detector evaluate --manifest corpus.json \
  --features work/extraction/extracted.json --output work/calibration
wobble-detector overlap --manifest corpus.json \
  --features work/extraction/extracted.json --output work/overlap.json
```

Each manifest example needs `id`, `source_id`, `label` (`too_wobbly` or
`not_too_wobbly`), `sha256`, and `video_path`. All variants of the same camera
recording must share `source_id` and, if supplied, `split_group`. Optional
`eligible_for_primary_eval: false` excludes an uncertain example. Each pair
needs `id`, `before_id`, `after_id`, and `repair_kind` (for example,
`same_interval_repair`). Overlap additionally needs each example's
`source_interval` with `start_sec` and `end_sec`, matching the export's duration
at original playback speed.

Extraction verifies frozen media hashes. Caches are tied to media bytes, code,
and extraction settings; evaluation rejects changed evidence. Source holdout
excludes every variant of the held-out recording from training. Overlap measures
only shared source time and does not invent new labels or independent examples.
These development commands update their output/cache directories; use separate
work directories for independent experiments. Only explicit human decisions
should supply labels. The private original corpus and browser review app are
not required or bundled.
