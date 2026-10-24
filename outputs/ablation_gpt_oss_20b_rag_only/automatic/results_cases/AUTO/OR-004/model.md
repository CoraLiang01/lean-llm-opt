**Mathematical Model**

```
Sets
    S   : set of distribution centers (suppliers)
    C   : set of customer groups

Parameters
    demand[c]          : daily demand of customer c          (file_0_view_0, column "demand")
    capacity[s]        : supply capacity of center s        (file_1_view_0, column "supply_capacity")
    cost[s,c]          : transportation cost per unit from s to c   (file_2_view_0, matrix entry)

Variables
    x[s,c]  ≥ 0        : quantity shipped from s to c

Objective
    minimize  Σ_{s∈S} Σ_{c∈C}  cost[s,c] * x[s,c]

Constraints
    /* Demand satisfaction */
    ∀ c∈C :  Σ_{s∈S} x[s,c]  =  demand[c]

    /* Supply capacity */
    ∀ s∈S :  Σ_{c∈C} x[s,c]  ≤  capacity[s]
```

**Data Mapping Rules**

| Parameter | Source Table | Source Column / Matrix |
|-----------|--------------|------------------------|
| `S`       | `file_1_view_0` | row identifier column `"Unnamed: 0"` |
| `C`       | `file_0_view_0` | row identifier column `"customer"` |
| `demand[c]` | `file_0_view_0` | column `"demand"` |
| `capacity[s]` | `file_1_view_0` | column `"supply_capacity"` |
| `cost[s,c]` | `file_2_view_0` | matrix entry at row `"Unnamed: 0"` = s, column `"C1"`‑`"C12"` = c |

*All indices `s` and `c` are taken from the corresponding identifier columns; the matrix in `file_2_view_0` is aligned exactly with these identifiers as verified by the observation.*