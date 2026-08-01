# Round 8 mouth-tracking ablation

This directory is the compact, auditable record of the 2026-07-31 mouth-only
tracking study for `20251225_K_1_2_FCMC` using
`round8_260729_snn031_m6`.

## Result

- Keep visible-node centroid tracking (`A0`) as the production method. The
  study did not establish a safe production replacement.
- Retain strict-mouth position-only tracking (`A1`) as the primary experimental
  candidate because it is the simplest mouth-only method that survived the
  study.
- Investigate the `A1`/`A2` assignment divergence near frame `114297` before
  reconsidering production behavior.
- Do not promote `A3`, `A4`, or `A5` from this study.

## Start here

| Artifact | Purpose |
| --- | --- |
| `report.html` | Portable reader-facing report with the principal results and evidence. |
| `COMPARISON_REPORT.md` | Concise comparison of the candidate methods. |
| `QUALITY_REVIEW.md` | Quality-control findings and stage-level interpretation. |
| `INDEPENDENT_AUDIT.md` | Detailed independent audit and limitations. |
| `FINAL_ARTIFACT_INDEX.json` | Canonical conclusions, source paths, and SHA-256 bindings. |
| `exact_commands.md` | Historical commands used to execute and audit the study. |
| `artifact.json` | Canonical source payload for the portable HTML report. |

The root JSON files preserve the cohort definition, input provenance, frozen
thresholds, body-length scales, protocol deviations, and the held-out schema
repair protocol. `quality_review/` contains compact independent-audit tables,
summaries, seals, and their generating scripts. The post-freeze directories
retain only stage preparation/completion/finalization records. Training sweep
directories retain only their compact completion summaries.

## Compact retention policy

On 2026-08-01, the report was reduced from a mixed report/workspace tree to a
compact scientific record. Candidate caches, per-view tracklets, 250-frame
hypothesis blocks, aggregated per-frame trajectories, parameter-run bodies,
quarantined copies, pytest/smoke outputs, and logs were removed because they are
regenerable execution artifacts rather than report content.

Before pruning, all 76 files referenced by `FINAL_ARTIFACT_INDEX.json` were
present and matched their recorded SHA-256 values; none of those indexed files
was removed. The historical commands and retained stage seals still document
the original execution. Future bulky reruns should use ignored `runs/` or
external output storage and copy only compact conclusions and provenance into
`report/`.

The packaged HTML passed payload validation and structural verification. Its
original delivery receipt records that interactive browser verification was not
run because a compatible local Chromium executable was unavailable.
