**Abstract Model**

```
SETS
    P   {1,…,80}   ! Products

PARAMETERS
    Demand[p]          ! Max demand (100 kg units) – file_0_view_0, column Ap, row 0
    Price[p]           ! Selling price ($/100 kg) – file_0_view_0, column Ap, row 1
    Cost[p]            ! Production cost ($/100 kg) – file_0_view_0, column Ap, row 2
    Quota[p]           ! Max units per day if all lines dedicated – file_0_view_0, column Ap, row 3
    ActCost[p]         ! Fixed activation cost ($) – file_1_view_0, column Ap, row 0
    BatchSize[p]       ! Minimum batch size (100 kg units) – file_2_view_0, column Ap, row 0
    M[p] = Demand[p]   ! Big‑M for binary activation

VARIABLES
    x[p]  integer >= 0   ! Units of product p produced (100 kg units)
    y[p]  binary         ! 1 if product p is activated, 0 otherwise

CONSTRAINTS
    1.  x[p] <= Demand[p]                     ∀p
    2.  x[p] <= 22 * Quota[p]                 ∀p
    3.  x[p] <= M[p] * y[p]                   ∀p
    4.  x[p] >= BatchSize[p] * y[p]           ∀p

OBJ:  maximize  Σ_p (Price[p] - Cost[p]) * x[p]  -  Σ_p ActCost[p] * y[p]
```

**Data Mapping**

| Parameter | Table ID | Column Name | Row |
|-----------|----------|-------------|-----|
| Demand[p] | file_0_view_0 | Ap | 0 |
| Price[p]  | file_0_view_0 | Ap | 1 |
| Cost[p]   | file_0_view_0 | Ap | 2 |
| Quota[p]  | file_0_view_0 | Ap | 3 |
| ActCost[p]| file_1_view_0 | Ap | 0 |
| BatchSize[p]| file_2_view_0 | Ap | 0 |

*All indices `p` correspond to the same column `Ap` across the tables (e.g., `A1` → product 1, `A2` → product 2, …, `A80` → product 80).*

This symbolic formulation captures the profit‑maximisation problem with integer production quantities, binary activation decisions, demand limits, daily quota limits, and minimum batch size requirements.