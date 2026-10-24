**Model**

```
SETS
    S   : source locations
    D   : destination locations

PARAMETERS
    c[i,j]   : unit transportation cost from source i to destination j
    supply[i]: available cargo units at source i
    demand[j]: required cargo units at destination j
    truck_cap: truck capacity (10 units)

VARIABLES
    t[i,j]   : integer number of trucks dispatched from i to j   (t[i,j] ∈ ℤ₊)
    y[i,j]   : continuous cargo units shipped from i to j      (y[i,j] ≥ 0)

MINIMIZE
    Z = Σ_{i∈S} Σ_{j∈D} c[i,j] * y[i,j]

CONSTRAINTS
    * Supply limits:
          Σ_{j∈D} y[i,j] ≤ supply[i]          ∀ i ∈ S

    * Demand satisfaction:
          Σ_{i∈S} y[i,j] = demand[j]          ∀ j ∈ D

    * Truck capacity coupling:
          y[i,j] ≤ truck_cap * t[i,j]          ∀ i ∈ S, j ∈ D
```

*Model type:* Mixed‑Integer Linear Program (MILP)  
*Objective sense:* minimize  
*Variable domains:* `t[i,j]` integer ≥ 0, `y[i,j]` continuous ≥ 0

---

**Data Mapping**

| Parameter / Set | Source Table | Key Column(s) | Value Column(s) | Notes |
|-----------------|--------------|---------------|-----------------|-------|
| `S`             | `file_2_view_0` | `source_id` | – | All 10 source IDs |
| `D`             | `file_1_view_0` | `destination_id` | – | All 20 destination IDs |
| `c[i,j]`        | `file_0_view_0` | Row: `source_id`<br>Column: `destination_id` | Cost values (`D1`…`D20`) | 10 × 20 matrix, fully populated |
| `supply[i]`     | `file_2_view_0` | `source_id` | `supply_units` | 10 supply values |
| `demand[j]`     | `file_1_view_0` | `destination_id` | `demand_units` | 20 demand values |
| `truck_cap`     | – | – | – | Constant value 10 |

All indices in the model (`i ∈ S`, `j ∈ D`) are taken directly from the corresponding tables; no additional filtering or aggregation is applied. The cost matrix uses the exact column names (`D1`…`D20`) as provided. The truck capacity is a fixed constant and not derived from data.