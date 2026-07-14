# Historical-video biological identity benchmark

This package compares idtracker.ai v6, native UDMT, and the repository's frozen
multi-view constrained contrastive linker on one identical historical-video
interval and one method-independent set of close-approach events.

## Truth hierarchy

1. **Primary:** a blinded human crossing audit. This is the only independent
   evidence about biological identity continuity.
2. **Identity-blind diagnostic:** each method's emitted points are matched to the
   unordered set of three top-view fish locations. This detects missing fish and
   off-animal drift without using identity labels.
3. **Geometry proxy:** after a global permutation, method labels are compared with
   provisional geometry labels. This is useful for diagnosis but is not biological
   ground truth and must not be presented as such.

The primary interval is inclusive frames 176300--181298 of
`20251225_K_1_2_FCMC`. Crossing events are selected exhaustively from pairwise
top-view proximity at 35 px, with at least two close frames and 83 frames of
available context on both sides. No method output participates in event selection.

## Method contracts

- **idtracker.ai v6:** official contrastive tracker, three animals, trained only
  from the locked cleaned top-view interval. Reflection suppression uses unordered
  3D candidate locations only as a foreground gate; it never supplies fish labels.
- **Native UDMT:** native XMem/ST-Net result on the same top-view interval. The
  primary predeclared variant is the trained Detectron2-binary seed because it was
  the strongest completed native result in the existing comparison.
- **Multi-view constrained contrastive:** a frozen encoder fit on disjoint earlier
  frames (train 12000--12999, calibration 10000--10499; evaluation here starts at
  176300). Same-frame geometry associates camera observations but its slot names
  are replaceable. Before/after prototypes are camera-balanced and a Hungarian
  one-to-one permutation propagates anonymous identities through each crossing
  episode. The current primary encoder is shared across cameras; top-vs-side
  encoders are a named ablation, not silently mixed into the primary result.

## Reproducible sequence

1. Produce or locate the three method trajectory artifacts.
2. Run `score_methods` to write the locked events and common diagnostic panel.
3. Run `render_blinded_audit` to create browser-compatible H.264 clips,
   `AUDIT_PLAYER.html`, and `AUDIT_WORKSHEET.csv`. All source frames are retained
   and stored at 60 fps so the high-speed recording is easier to inspect.
4. Have an auditor use the player or fill the worksheet without opening
   `PRIVATE_BLINDING_KEY.csv`.
5. Run `summarize_audit` to join ratings to the private key and report
   pass/switch/uncertain counts, Wilson intervals, and paired exact comparisons,
   retaining localization failures as failures rather than dropping them.
6. Run `build_final_report` to apply the pre-audit decision rule and package the
   final source-backed report. For an older `mp4v` audit bundle, run
   `repair_audit_clip_codecs` once; it verifies codec, dimensions, and exact frame
   counts before atomically replacing each clip.

All reusable scripts take paths as CLI arguments. Machine-specific data roots are
not embedded in source code.
