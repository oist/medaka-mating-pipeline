# Round8 mouth-method comparison GUI

`scripts.visualization.round8_mouth_method_gui` is a read-only three-camera
viewer for the controlled Round8 comparison on
`20251225_K_1_2_FCMC`.

It shows synchronized source frames from all three cameras and lets the reviewer
switch the identity overlay among:

- **A0** — visible-node-centroid production reference;
- **A1** — strict-mouth position-only relinking;
- **A2** — strict-mouth relinking with robust velocity; and
- **A4** — conservative reciprocal tracklet paths.

The background prediction overlay comes from the authenticated Round8 candidate
cache. It contains up to six mouth-valid SLEAP candidates per frame and preserves
each candidate's original SLEAP source-instance index. Candidate text is hidden
by default to keep the fish visible. If **Raw candidate indices** is enabled,
`c2/raw4` means candidate-cache slot 2, originating from raw SLEAP instance 4.
Rare raw instances with no finite mouth are not present in this strict-mouth
cache.

## Start

From the repository root:

```powershell
.venv\Scripts\python.exe -m scripts.visualization.round8_mouth_method_gui `
  20251225_K_1_2_FCMC
```

The default command resolves:

- videos and the unchanged A0 result through `configs/paths.local.toml`;
- A1/A2/A4 and the authenticated candidate caches through
  `report/20260731/round8_mouth_tracking_ablation/`;
- the fixed refractive calibration through the local paths config; and
- female evidence through the independent comparison output and authoritative
  annotation ledger.

To open the reported `34547→34548` event directly:

```powershell
.venv\Scripts\python.exe -m scripts.visualization.round8_mouth_method_gui `
  20251225_K_1_2_FCMC --frame 34548 --context-frames 500
```

Validate all resolved inputs without opening Qt:

```powershell
.venv\Scripts\python.exe -m scripts.visualization.round8_mouth_method_gui `
  20251225_K_1_2_FCMC --check-only
```

If a mapped drive is unavailable in the launching shell, pass an accessible
recording root and A0 CSV explicitly:

```powershell
.venv\Scripts\python.exe -m scripts.visualization.round8_mouth_method_gui `
  20251225_K_1_2_FCMC `
  --video-dir <recording-root>\20251225_K_1_2_FCMC `
  --a0-csv <round8-output>\round8_260729_snn031_m6_3d_identified.csv
```

## Reading the overlays

The default view is intentionally sparse:

- A tiny gray filled dot is a raw predicted mouth.
- A tiny gray open ring is a raw predicted tail; a 1-pixel gray line connects
  the two nodes.
- A thin colored pose is the exact candidate slot recorded by A1/A2/A4, or the
  nearest anchor-compatible candidate for A0.
- A small colored cross is the A0 reprojected visible-node centroid. A small
  colored ring is the A1/A2/A4 reprojected mouth position.
- Color is stable for a method's base identity. Identity text is off by default;
  the right-side table remains the uncluttered source for ID and segment names.
- A small magenta star appears only on an exact confirmed female-annotation
  frame. It is never propagated to unannotated frames.
- Each camera pane supports mouse-wheel zoom and drag-to-pan. `F` fits all
  panes back to the window.

Raw mouths, raw tail/link predictions, the selected pose, method anchors,
candidate indices, identity/GT text, trails, and the female marker can each be
switched independently. Trails and all on-image text start off. **Marker
radius** changes only the colored selected-pose and method-anchor glyphs.

The right-side table reports every method row at the current frame, including
unresolved rows, segment, candidate slots, and residual. The female evidence
line shows the preceding and following independently mapped female annotations;
it does not interpolate the female between them.

## Female-change review queue

The event menu contains each observed change in the independently evaluated
female assignment sequence for A0, A1, and A2. For a transition such as
`11 → 8`, the GUI searches the two method identities between the bracketing
female annotations and centers the review window on their minimum 3D distance.
This supplies a concrete close-contact candidate while preserving the original
annotation bracket.

The queue also contains the reported `34547→34548` descriptive event and every
final bounded A4 evaluation window.

## A4 limitation

A4 was excluded from the full-length finalists and was only run on registered
bounded windows. The GUI merges the final critical, validation, and held-out A4
outputs. Outside those windows it reports no rows. Within a window, blank
`PATH_*` rows are explicit A4 abstentions, not a viewer error.

Consequently, A4 cannot be used to claim a full-recording female identity or to
compare every A0/A1/A2 female transition. It can still be inspected as the
high-purity, high-abstention diagnostic that was actually evaluated.

## Controls

- `0`, `1`, `2`, `4`: switch method.
- `Space`: play/pause.
- `Left` / `Right`: one source frame.
- `Shift+Left` / `Shift+Right`: ten source frames.
- `F`: fit all camera panes.
- Overlay checkboxes: reveal only the prediction or identity detail needed for
  the current frame.
- Event buttons/menu: move among detected female-assignment changes and bounded
  diagnostic windows.
- Exact-frame box: jump to any source frame; the local scrub range recenters
  automatically.

The GUI does not write tracking, annotations, videos, SLP files, or report
artifacts.
