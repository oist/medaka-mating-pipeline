# Optimized baseline report

## Outcome

The optimized window/checkpoint implementation completed the full 348,600-frame
recording in 575.555 s (9.59 min) from warm local candidate caches, with 3.61 GB
peak process-tree RSS and zero unresolved seams. It is an engineering success
and a scientific rejection: its identity stability and direct coverage are
materially worse than the historical strict-mouth output.

## Frozen configuration

The run uses experiment `20251225_K_1_2_FCMC`, round `round7_260704`, fixed
refractive calibration (`mm^2` ray residual), strict finite mouth, six candidate
slots per view, float64 candidate coordinates, XYZ bounds 0--200 mm, 8 mm/frame
motion gate, 80 mm hard cap, 100-frame gap limit, 10,000-frame cores, 250-frame
context, eight CPU workers, framewise legacy hypothesis selection, no explicit
fragment links, no tail term, sequential-Hungarian/restart plumbing, and seed
20260717. `resolved_configuration.json` records all code, calibration, and cache
hashes.

## Optimization work

- Each `.slp` is decoded once into an HDF5 view cache; later runs read only the
  requested frame slice. Cold decode/write time was 330.409 s total for the
  three views. Cache sizes are 31.4--32.0 MB and all hashes are recorded.
- Independent ownership cores enable bounded-memory parallel execution and
  atomic checkpoint reuse. Exact resource-based seam reconciliation prevents
  duplicate ownership.
- The hot framewise tracker uses compact NumPy arrays and Hungarian assignment;
  the array implementation was exact-value equivalent to the reference on a
  1,000-frame comparison while reducing that kernel to about 0.06 s.
- Compact no-link fragment summaries avoid the earlier dataframe finalizer
  bottleneck. The final full run writes directly from reconciled arrays.
- Eight workers were chosen from the measured 1/2/4/8/16 sweep; 16 was slower
  and offered no quality difference.

The RTX 5070 Ti was evaluated but not used end-to-end. On real endpoint data,
one million float64 endpoint computations took 0.07979 s on CPU, 0.000494 s as
a synchronized CUDA kernel, and 0.008550 s including host transfer. A 250,000
batch of 3x3 solves took 0.06498 s on CPU versus 0.000990 s as a CUDA kernel.
Maximum CPU/GPU differences were 1.78e-15 and 2.13e-14. The pipeline remains
dominated by cache/window orchestration, triangulation setup, merge, and output;
a GPU port was therefore not introduced without an end-to-end win. GPU
utilization and peak VRAM were not sampled and are explicitly unavailable.

## Runtime and artifacts

The final run produced 35 checkpoints, 1,030,486 direct observations, 92,426
fragments, 3,575 alias candidates, 618 accepted aliases, no unresolved seam,
and 1,045,800 filled rows. The final CSV is 85,984,583 bytes with SHA-256
`340078e8e2c6c09c8177fd717daf44a054faad116380f83f65bbbd8b34de5d0b`.

Fragmentation is severe and is another rejection signal: fragment length p50,
p95, p99, and maximum are 1, 19, 229, and 5,280 observed frames; 78,222 of
92,426 fragments (84.63%) are singletons. Direct mouth residual p50/p95/p99 is
0.229/1,680.002/4,691.415 mm^2 because this baseline has no residual hard gate;
direct tail residual p50/p95/p99 is 0.328/1.813/5.294 mm^2. Resource conflicts
and frames with more than three active trajectories are both zero. Link
precision cannot be measured without all-fish link truth; accepted-alias and
resource-conflict counts are proxies, not precision labels.

The final algorithm is faster than the previously documented 15.6-minute
strict-mouth full run by approximately 38.5% (1.63x), comparing warm scientific
runs. Including the one-time cold cache build gives about 906.0 s, so caching
pays off through reuse rather than making the first invocation faster.

## Scientific comparison

| Metric | Historical strict mouth | Optimized candidate |
|---|---:|---:|
| Female GT eligible / mapped | 300 / 298 | 300 / 283 |
| Best fixed female accuracy | 0.5235 | 0.4276 |
| Observed female-ID changes | 11 | 81 |
| Strict female accuracy | 0.5187 | 0.4191 |
| All-three finite frames | 0.9898 | 0.9689 |
| All-three direct frames | 0.9816 | 0.8943 |
| Jump p95 / p99 (mm) | 1.134 / 2.202 | 1.485 / 7.310 |
| Adjacent jumps above 8 mm | 0 | 0 |
| Reviewed events preserved / evaluable | 3 / 4 | 3 / 3 |

On the common 283 mapped female rows, accuracy is 0.5300 historical versus
0.4276 optimized and observed changes are 11 versus 81. Cohort-level metrics
show the same direct-coverage and p99-jump regression in all four event windows
and most controls. The optimized candidate's hard cap prevents jumps above 8
mm, but many transitions accumulate near the gate; passing the hard gate alone
does not establish identity quality.

## Decision

Do not replace production output with the optimized candidate. Retain the
candidate caches, bounded windows, checkpoint/resume, provenance, exact seam
reconciliation, and array kernels as reusable engineering infrastructure.
Before scientific promotion, the framewise/window identity path must reproduce
historical direct coverage and female stability on identical inputs, then be
tested with ground truth for all three fish. The current GUI ledger labels only
the female and cannot independently detect male-to-male swaps.
