"""Local-file adapter for the unchanged calibrated extractor.

The original workspace redirected paths to a media volume. This standalone
package reads only the path supplied by its caller; it has no catalog or mount
configuration. Keeping this adapter preserves the calibrated extractor's bytes.
"""
from pathlib import Path


def media_filename(path):
    return str(Path(path).expanduser())
