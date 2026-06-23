"""Cross-cutting experimental harnesses that compare pipeline methods.

Unlike the per-stage packages (``scripts/tracking``, ``scripts/reconstruction_3d``,
...), modules here orchestrate *several* existing methods on shared inputs and
score them with a common metric panel. They produce inspectable, reproducible
results directories rather than production tracks.
"""
