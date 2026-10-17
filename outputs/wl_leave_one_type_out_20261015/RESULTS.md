# WL similarity with leave-one-type-out reference matching

Setting: cumulative rounds 0–2; effective bounds retained as variable restrictions and excluded from the graph.

For each test P, exclude every reference with the same category as P, then take the maximum remaining similarity. The reported LOTO column is the median of these per-test maxima, not an accuracy measure.

## large_scale_101

| Category | Tests | Median same | Mean same | Median all | Median LOTO |
|---|---:|---:|---:|---:|---:|
| NRM | 25 | 0.3234 | 0.3234 | 0.3330 | 0.3330 |
| RA | 22 | 0.6562 | 0.6564 | 0.6562 | 0.6500 |
| TP | 9 | 0.6240 | 0.6314 | 0.6343 | 0.6327 |
| AP | 5 | 0.0508 | 0.0551 | 0.4424 | 0.4424 |
| UFLP | 14 | 0.5503 | 0.5557 | 0.5503 | 0.4446 |
| Mixture | 18 | 0.3983 | 0.4499 | 0.5025 | 0.4057 |
| Others | 8 | 0.2584 | 0.3025 | 0.3811 | 0.3615 |
| All | 101 | 0.5065 | 0.4632 | 0.5340 | 0.4424 |

## variants

| Category | Tests | Median same | Mean same | Median all | Median LOTO |
|---|---:|---:|---:|---:|---:|
| AP | 3 | 0.3569 | 0.3582 | 0.3569 | 0.2749 |
| UFLP | 3 | 0.1981 | 0.1972 | 0.3078 | 0.3078 |
| Mixture | 17 | 0.6665 | 0.6157 | 0.6665 | 0.2520 |
| Others | 13 | 0.2373 | 0.2993 | 0.4359 | 0.4281 |
| All | 36 | 0.4325 | 0.4451 | 0.5588 | 0.3127 |

## Input verification

{'unchanged': 155}

NRM uses the previously revised reference. Its LOTO score excludes Ref03 entirely. AP variable-domain differences and omitted numerical attributes remain limitations. No high/low similarity threshold is introduced.
