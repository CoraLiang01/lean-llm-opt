# Grouped LOTO WL results

Primary setting: h=2, effective bounds retained as variable restrictions and excluded from the graph.
For Mixture and Others tests, all 8 Mixture and 2 Others references are removed jointly. For NRM, RA, TP, AP and UFLP, only the test category references are removed.

## large_scale_101

| Category | Tests | Exclusion rule | Median grouped LOTO | Mean grouped LOTO |
|---|---:|---|---:|---:|
| NRM | 25 | same category excluded | 0.3330 | 0.3330 |
| RA | 22 | same category excluded | 0.6500 | 0.6505 |
| TP | 9 | same category excluded | 0.6327 | 0.6369 |
| AP | 5 | same category excluded | 0.4424 | 0.4437 |
| UFLP | 14 | same category excluded | 0.4446 | 0.4493 |
| Mixture | 18 | Mixture+Others jointly excluded | 0.3458 | 0.4027 |
| Others | 8 | Mixture+Others jointly excluded | 0.3298 | 0.3815 |
| All | 101 | Mixture+Others jointly excluded for Mixture/Others; same category otherwise | 0.4401 | 0.4671 |

## variants

| Category | Tests | Exclusion rule | Median grouped LOTO | Mean grouped LOTO |
|---|---:|---|---:|---:|
| AP | 3 | same category excluded | 0.2749 | 0.2752 |
| UFLP | 3 | same category excluded | 0.3078 | 0.3079 |
| Mixture | 17 | Mixture+Others jointly excluded | 0.2393 | 0.3211 |
| Others | 13 | Mixture+Others jointly excluded | 0.3358 | 0.3520 |
| All | 36 | Mixture+Others jointly excluded for Mixture/Others; same category otherwise | 0.3098 | 0.3273 |

