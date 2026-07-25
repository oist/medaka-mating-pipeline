# Side top-down optimization: shared-benchmark evaluation

This is diagnostic selection/regression evidence, not a genuinely locked generalization test. The pilot is Round-8-held-out but Round-7-exposed; the stress cohort is in-sample.

## centroid only

### multi_experiment_in_sample_stress

| candidate | frames | precision | recall | all 3 | mouth p95 | tail p95 | swaps | shared tails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centroid_sigma25 | 61 | 0.9944 | 0.9727 | 0.9180 | 3.5335 | nan | nan | 0.0000 |
| centroid_sigma20 | 61 | 1.0000 | 0.9781 | 0.9344 | 3.1893 | nan | nan | 0.0000 |
| centroid_sigma15 | 61 | 1.0000 | 0.9727 | 0.9180 | 3.8399 | nan | nan | 0.0000 |

Fully labeled instances only:

| candidate | matched / GT | mouth p95 | mouth p99 | tail p95 | tail p99 |
|---|---:|---:|---:|---:|---:|
| centroid_sigma25 | 168 / 173 | 3.6703 | 5.5043 | nan | nan |
| centroid_sigma20 | 169 / 173 | 3.2413 | 5.3656 | nan | nan |
| centroid_sigma15 | 168 / 173 | 4.0675 | 7.4479 | nan | nan |

### round8_pilot_event_holdout

| candidate | frames | precision | recall | all 3 | mouth p95 | tail p95 | swaps | shared tails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centroid_sigma25 | 29 | 1.0000 | 0.9885 | 0.9655 | 5.9249 | nan | nan | 0.0000 |
| centroid_sigma20 | 29 | 1.0000 | 0.9885 | 0.9655 | 5.8703 | nan | nan | 0.0000 |
| centroid_sigma15 | 29 | 1.0000 | 1.0000 | 1.0000 | 5.2181 | nan | nan | 0.0000 |

Fully labeled instances only:

| candidate | matched / GT | mouth p95 | mouth p99 | tail p95 | tail p99 |
|---|---:|---:|---:|---:|---:|
| centroid_sigma25 | 86 / 87 | 5.9249 | 8.6247 | nan | nan |
| centroid_sigma20 | 86 / 87 | 5.8703 | 21.0086 | nan | nan |
| centroid_sigma15 | 87 / 87 | 5.2181 | 9.0059 | nan | nan |

## oracle centered instance

### multi_experiment_in_sample_stress

| candidate | frames | precision | recall | all 3 | mouth p95 | tail p95 | swaps | shared tails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| instance_control | 61 | 1.0000 | 1.0000 | 1.0000 | 0.6483 | 7.5644 | 0.0000 | 0.1148 |
| instance_jitter8 | 61 | 1.0000 | 1.0000 | 1.0000 | 5.6255 | 9.5714 | 0.0000 | 0.1148 |
| instance_jitter16 | 61 | 1.0000 | 1.0000 | 1.0000 | 9.9658 | 29.4169 | 0.0000 | 0.2295 |

Fully labeled instances only:

| candidate | matched / GT | mouth p95 | mouth p99 | tail p95 | tail p99 |
|---|---:|---:|---:|---:|---:|
| instance_control | 173 / 173 | 0.6485 | 0.6941 | 7.5644 | 52.9930 |
| instance_jitter8 | 173 / 173 | 5.6579 | 7.0293 | 9.5714 | 60.7667 |
| instance_jitter16 | 173 / 173 | 10.3274 | 13.1805 | 29.4169 | 136.6154 |

### round8_pilot_event_holdout

| candidate | frames | precision | recall | all 3 | mouth p95 | tail p95 | swaps | shared tails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| instance_control | 29 | 1.0000 | 1.0000 | 1.0000 | 0.6771 | 3.5761 | 0.0000 | 0.0000 |
| instance_jitter8 | 29 | 1.0000 | 1.0000 | 1.0000 | 4.4717 | 4.7254 | 0.0000 | 0.0000 |
| instance_jitter16 | 29 | 1.0000 | 1.0000 | 1.0000 | 3.9216 | 4.9910 | 0.0000 | 0.0000 |

Fully labeled instances only:

| candidate | matched / GT | mouth p95 | mouth p99 | tail p95 | tail p99 |
|---|---:|---:|---:|---:|---:|
| instance_control | 87 / 87 | 0.6771 | 0.7342 | 3.5761 | 6.9089 |
| instance_jitter8 | 87 / 87 | 4.4717 | 5.5335 | 4.7254 | 8.5198 |
| instance_jitter16 | 87 / 87 | 3.9216 | 6.2601 | 4.9910 | 9.1105 |

## full topdown grid

### multi_experiment_in_sample_stress

| candidate | frames | precision | recall | all 3 | mouth p95 | tail p95 | swaps | shared tails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sigma25__control | 82 | 0.9871 | 0.9312 | 0.8025 | 3.9108 | 9.5384 | 0.0000 | 0.0864 |
| sigma25__jitter8 | 82 | 0.9828 | 0.9271 | 0.7901 | 5.5934 | 10.6669 | 0.0000 | 0.0741 |
| sigma25__jitter16 | 82 | 0.9871 | 0.9312 | 0.8025 | 7.5395 | 23.6854 | 0.0000 | 0.1728 |
| sigma20__control | 82 | 0.9957 | 0.9312 | 0.8025 | 3.5287 | 10.3775 | 0.0000 | 0.0864 |
| sigma20__jitter8 | 82 | 0.9957 | 0.9312 | 0.8025 | 5.0050 | 10.2072 | 0.0000 | 0.0864 |
| sigma20__jitter16 | 82 | 0.9957 | 0.9312 | 0.8025 | 8.0865 | 29.0765 | 0.0000 | 0.1852 |
| sigma15__control | 82 | 0.9914 | 0.9312 | 0.7901 | 3.6840 | 10.1297 | 0.0000 | 0.0976 |
| sigma15__jitter8 | 82 | 0.9870 | 0.9231 | 0.7778 | 5.4296 | 9.4508 | 0.0000 | 0.0617 |
| sigma15__jitter16 | 82 | 0.9914 | 0.9312 | 0.7901 | 7.9483 | 15.0772 | 0.0000 | 0.1341 |

Fully labeled instances only:

| candidate | matched / GT | mouth p95 | mouth p99 | tail p95 | tail p99 |
|---|---:|---:|---:|---:|---:|
| sigma25__control | 207 / 213 | 4.1020 | 6.4248 | 6.8963 | 25.1703 |
| sigma25__jitter8 | 207 / 213 | 5.7585 | 8.2772 | 6.7635 | 20.9444 |
| sigma25__jitter16 | 207 / 213 | 7.6422 | 14.9343 | 13.8076 | 58.3644 |
| sigma20__control | 207 / 213 | 3.5483 | 5.7077 | 7.1537 | 25.0092 |
| sigma20__jitter8 | 208 / 213 | 5.3437 | 9.3842 | 6.7121 | 20.7744 |
| sigma20__jitter16 | 208 / 213 | 8.0415 | 15.3838 | 16.1768 | 58.4697 |
| sigma15__control | 206 / 213 | 4.0928 | 6.2978 | 9.7130 | 24.7797 |
| sigma15__jitter8 | 207 / 213 | 5.5290 | 11.4944 | 6.7291 | 16.4024 |
| sigma15__jitter16 | 208 / 213 | 7.9285 | 17.9674 | 9.6481 | 51.5281 |

### round8_pilot_event_holdout

| candidate | frames | precision | recall | all 3 | mouth p95 | tail p95 | swaps | shared tails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sigma25__control | 40 | 1.0000 | 0.9250 | 0.7750 | 5.8724 | 6.8522 | 0.0000 | 0.0000 |
| sigma25__jitter8 | 40 | 1.0000 | 0.9250 | 0.7750 | 4.6609 | 6.6704 | 0.0000 | 0.0250 |
| sigma25__jitter16 | 40 | 1.0000 | 0.9250 | 0.7750 | 4.2667 | 7.5451 | 0.0000 | 0.0250 |
| sigma20__control | 40 | 1.0000 | 0.9250 | 0.7750 | 6.0502 | 7.1959 | 0.0000 | 0.0250 |
| sigma20__jitter8 | 40 | 1.0000 | 0.9250 | 0.7750 | 5.2191 | 5.7512 | 0.0000 | 0.0250 |
| sigma20__jitter16 | 40 | 1.0000 | 0.9250 | 0.7750 | 4.2779 | 7.5744 | 0.0000 | 0.0250 |
| sigma15__control | 40 | 1.0000 | 0.9500 | 0.8500 | 6.4334 | 8.3740 | 0.0000 | 0.0000 |
| sigma15__jitter8 | 40 | 1.0000 | 0.9500 | 0.8500 | 8.5249 | 8.0277 | 0.0000 | 0.0000 |
| sigma15__jitter16 | 40 | 1.0000 | 0.9500 | 0.8500 | 7.5261 | 8.1371 | 0.0000 | 0.0250 |

Fully labeled instances only:

| candidate | matched / GT | mouth p95 | mouth p99 | tail p95 | tail p99 |
|---|---:|---:|---:|---:|---:|
| sigma25__control | 107 / 108 | 5.8724 | 7.9226 | 5.5752 | 8.0637 |
| sigma25__jitter8 | 107 / 108 | 4.6609 | 5.4375 | 4.8126 | 9.2207 |
| sigma25__jitter16 | 107 / 108 | 4.2667 | 9.7318 | 4.8354 | 10.2512 |
| sigma20__control | 106 / 108 | 6.0502 | 53.0773 | 4.0400 | 8.1624 |
| sigma20__jitter8 | 107 / 108 | 5.2191 | 48.0370 | 4.7181 | 103.5442 |
| sigma20__jitter16 | 107 / 108 | 4.2779 | 49.4362 | 4.7233 | 104.8727 |
| sigma15__control | 107 / 108 | 6.4334 | 47.0721 | 3.9364 | 7.9610 |
| sigma15__jitter8 | 108 / 108 | 8.5249 | 50.5273 | 5.7956 | 8.2365 |
| sigma15__jitter16 | 108 / 108 | 7.5261 | 50.9289 | 6.4249 | 94.3412 |
