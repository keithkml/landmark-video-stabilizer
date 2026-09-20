"""Render the composed correction directly from the original, in RGB16."""

import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .core import native_matrix, require, sampling_map


def _filter_filename(path):
    # FFmpeg's filter grammar has two escaping levels; simple paths avoid
    # ambiguous quoting. A local symlink can give a LUT a simple pathname.
    value = str(Path(path).resolve())
    require(not any(c in value for c in "'\\:[],;\n\r"), "Use a LUT path without filter syntax characters")
    return value


def render(recipe, plan_dir, source, output, lut=None, audio=None, limit_frames=None, ffmpeg="ffmpeg"):
    source, output, plan_dir = Path(source).resolve(), Path(output).resolve(), Path(plan_dir).resolve()
    require(source.is_file(), "Source video does not exist")
    require(output.suffix.lower() == ".mp4", "Output must be an MP4")
    require(not output.exists(), "Refusing to overwrite output")
    require(output.parent.is_dir(), "Output directory does not exist")
    require(shutil.which(ffmpeg) is not None, "FFmpeg is required on PATH")
    settings = recipe["render"]
    matrices = np.load(plan_dir / "affine.npy", allow_pickle=False)
    fields = np.load(plan_dir / "residual-fields.npy", mmap_mode="r", allow_pickle=False)
    first, end = settings["source_frames"]
    require(matrices.shape == (end - first, 2, 3) and len(fields) == len(matrices), "Plan frame count mismatch")
    count = len(matrices) if limit_frames is None else limit_frames
    require(0 < count <= len(matrices), "Invalid requested frame count")
    crop = settings["crop"]
    width, height = crop[2:]
    sw, sh = settings["source_display_size"]
    require([width, height] == recipe["residual"]["output_size"], "Crop and residual dimensions disagree")
    require(width % 2 == 0 and height % 2 == 0, "H.264 4:2:0 dimensions must be even")
    require(Fraction(settings["fps"]) > 0, "Invalid frame rate")
    filters = settings.get("color_filter", "null")
    if "{lut}" in filters:
        require(lut is not None and Path(lut).is_file(), "This recipe requires its original LUT via --lut")
        filters = filters.replace("{lut}", _filter_filename(lut))
    vf = f"trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS,{filters},format=rgb48le"
    decode = [ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-vf", vf, "-an",
              "-threads", "4", "-filter_threads", "4", "-fps_mode", "passthrough", "-frames:v", str(count),
              "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    temporary = output.with_name(output.stem + ".rendering.mp4")
    record = output.with_suffix(".render.json")
    require(not temporary.exists() and not record.exists(), "Render artifacts already exist")
    encode = [ffmpeg, "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb48le",
              "-s", f"{width}x{height}", "-r", settings["fps"], "-i", "-"]
    if audio is not None:
        require(Path(audio).is_file(), "Audio file does not exist")
        # Caller supplies audio already cut and aligned to the selected passage.
        encode += ["-i", str(Path(audio).resolve()), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        encode += ["-an"]
    encode += ["-vf", "scale=iw:ih:in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p",
               "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-maxrate", "24M", "-bufsize", "48M",
               "-threads", "4", "-filter_threads", "4", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-colorspace", "bt709", "-color_range", "tv", "-movie_timescale", "48000",
               "-t", str(float(Fraction(count, 1) / Fraction(settings["fps"]))),
               "-movflags", "+faststart", str(temporary)]
    # Check EVERY frame before starting expensive rendering. No fabricated borders.
    native = [native_matrix(a, settings["analysis_width"], sw, crop) for a in matrices[:count]]
    for i, matrix in enumerate(native):
        try:
            sampling_map(matrix, fields[i], (width, height), (sw, sh))
        except ValueError as error:
            raise ValueError(f"Source bounds failed at output frame {i}: {error}") from error
    logs = [output.with_suffix(".decode.log"), output.with_suffix(".encode.log")]
    require(not any(path.exists() for path in logs), "Render logs already exist")
    with logs[0].open("x") as decoder_log, logs[1].open("x") as encoder_log:
        decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=decoder_log)
        encoder = None
        try:
            encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=encoder_log)
            for i, matrix in enumerate(native):
                chunk = decoder.stdout.read(sw * sh * 6)
                require(len(chunk) == sw * sh * 6, f"Incomplete source frame {i}; inspect decode log")
                frame = np.frombuffer(chunk, np.uint16).reshape(sh, sw, 3)
                mx, my = sampling_map(matrix, fields[i], (width, height), (sw, sh))
                encoder.stdin.write(cv2.remap(frame, mx, my, cv2.INTER_CUBIC).tobytes())
            encoder.stdin.close()
            require(not decoder.stdout.read(1), "Unexpected extra decoded data")
            decoder.stdout.close()
            require(decoder.wait() == 0 and encoder.wait() == 0, "FFmpeg failed; inspect logs")
        finally:
            for process in (decoder, encoder):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
    # Full decode is checked before exposing the final filename.
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(temporary),
                    "-f", "null", "-"], check=True, capture_output=True)
    require(not output.exists(), "Output appeared while rendering")
    temporary.rename(output)
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"output": str(output), "sha256": digest, "frames": count,
              "source_frames": [first, first + count], "decode": decode, "encode": encode,
              "audio": "supplied aligned audio stream copied" if audio else "video only",
              "spatial_resampling_passes": 1, "full_decode_passed": True,
              "continuous_motion_verified": False, "listening_verified": False, "quality_approved": False}
    record.write_text(json.dumps(result, indent=2) + "\n")
    return result
