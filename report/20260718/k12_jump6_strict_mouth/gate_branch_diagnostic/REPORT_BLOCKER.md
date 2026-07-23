# Portable report packaging blocker

The canonical report payload was generated as `artifact.json`, and local
structural checks passed for block order, datasets, source IDs, chart fields,
table sorts, and row counts.

The required portable builder could not create `report.html` because this
machine has no available `node`, `nodejs`, `bun`, or `deno` executable. The
attempted command was the Data Analytics plugin's
`deliver_portable_artifact.mjs` workflow. No alternate hand-authored HTML was
created because that would bypass the required canonical renderer and QA.

This packaging limitation does not affect the diagnostic CSVs or the validated
female-annotator queue.
