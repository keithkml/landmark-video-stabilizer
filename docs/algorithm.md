# Method, coordinates, and evidence

## Two corrections

1. Track stationary bark against a fixed reference frame. Reserve landmarks from
   fitting and estimate one full affine transform per frame using robust
   inliers. Full affine permits translation, rotation, unequal scale, and shear;
   it is not a translation-only or rigid-only correction.
2. Measure remaining movement in the globally stabilized export. Fit a small,
   smooth displacement field in explicitly selected background regions. Blend
   the field to zero around the bird and foreground trunk.
3. Compose the field and the affine mapping. Apply their combined output-to-source
   map directly to each original RGB16 camera frame with cubic interpolation.
   There is no temporal interpolation, invented background, or frozen subject.

The same camera movement can produce different apparent motion at different
depths. A whole-image affine transform can lock the near trunk while leaving
the distant branch moving. Separate, bounded regional corrections addressed
that residual in the woodpecker recording. The recording alone does not establish
whether optical stabilization, rolling shutter, or camera translation caused
each observed deformation.

## Residual fit

The input is displacement from fixed-reference landmark locations in the baseline
export. Well-tracked background landmarks are selected by their valid-frame
fraction. Missing samples of selected tracks are linearly interpolated, including
constant extension at the ends; this is appropriate only for rare tracking gaps.
Validity masks remain in use when scoring consecutive-frame residuals.

Gaussian basis functions centered on training landmarks form a regularized least
squares fit for each frame. The example evaluates widths 80, 140, 220, and 320
output pixels and chooses the best maximum adjacent-frame residual among reserved
validation landmarks. Width 320 won for the woodpecker. The chosen fields are
sampled on a 65 x 65 grid and upsampled to native output resolution during render.

The reserved landmarks are excluded from coefficient fitting, **but used for model
selection**. Their errors are validation scores, not an untouched final test set.
Independent measurement of the encoded export and actual viewing are separate
steps. Even exported feature tracking can fail, so confidence and worst frames
must be examined along with displacement numbers.

## Data contract

All NumPy files are loaded with `allow_pickle=False`.

`tracks` is an NPZ with:

| Key | Shape | Meaning |
| --- | --- | --- |
| `reference_points` | P x 2 | Landmark x,y positions in native output pixels |
| `displacement` | N x P x 2 | Current position minus reference position |
| `valid` | N x P | Boolean tracking validity |
| `protected_count` | scalar | Leading foreground landmarks excluded from the background fit |

The historical fixture uses the equivalent name `bark_count` for
`protected_count`. The remaining landmarks are candidate background points.

`affine` is an N x 2 x 3 NPY: a fixed-reference-to-current-source map in analysis
coordinates. Rendering converts it to an output-to-source native-pixel mapping.
The analysis image must preserve the source display aspect ratio. If the native
width is `s` times the analysis width, pixel centers follow
`native = s * analysis + (s - 1) / 2`. The renderer accounts for this offset and
the fixed crop origin. Stored camera dimensions and display dimensions can differ
when rotation metadata is present; the example is an upright portrait source.

`render.source_frames` is `[start, end)` in original-frame indices. Matrix index
zero corresponds to `start`; frame rate remains the original rational rate.
`crop` is `[left, top, width, height]` in native reference coordinates.

## Correction regions

Each region is a product of smooth ramps; the final mask is the maximum over
regions. `x_fall: [a,b]` is one at x <= a, zero at x >= b, and a cubic smoothstep
between. `x_rise`, `y_fall`, and `y_rise` use the corresponding direction. The
woodpecker recipe combines a left background region with an upper-left region.

Leave a generous zero-mask margin around every observed bird position, accounting
for grid interpolation support. The bird still receives the global affine
stabilization; only the additional background warp is excluded there.

## Checks and limitations

- The example limits grid-map Jacobians to (0.8, 1.2) and correction magnitude to
  less than 25 pixels. Its actual maximum correction is 8.88 pixels. These bounds
  do not prove that local geometry is natural or that the native interpolated
  map is perceptually acceptable.
- Native sampling bounds are checked for every output pixel in every frame, with
  source support for cubic interpolation. A crop that needs nonexistent source
  pixels is rejected before rendering.
- The renderer uses a fixed output size, no temporal synthesis, and one spatial
  resampling operation. It preserves the chosen color filter in RGB16 until
  BT.709 limited-range 4:2:0 encoding.
- Export measurements report all landmarks and separately identify those with a
  full 41-pixel margin for the whole recording. Edge exclusion is based on
  position and support, not on the magnitude of a tracking error.
- A new recipe needs scene inspection and motion/audio review. Numerical success
  is not an approval gate by itself.

## Original successful export

The saved correction covers 2,910 frames, 48.5485 seconds, at 60000/1001 fps in a
1240 x 1240 crop. The original encoded result's measured maximum adjacent-frame
trunk movement was 0.370 pixels. For 129 interior landmarks the maximum was
0.982 pixels, with minimum per-frame valid fraction 0.961. Including seven edge
locations, the maximum was 1.071 pixels. Some slow background displacement remains.

All frames were inspected as contact sheets, with 52 larger samples and four
native-resolution frames. Continuous perceptual playback and audio listening by
the assistant remained unverified; the user subsequently responded positively
to this exact export. These are historical measurements, not guarantees for
future footage or newly encoded versions.
