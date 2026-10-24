**Mathematical Model (Facility‑Location Problem)**  

```
Sets
    I   : {S1, S2}          (suppliers)
    J   : {C1, C2}          (supermarkets)

Parameters
    fixed_cost[i]   : fixed activation cost of supplier i
    transport_cost[i,j] : unit transportation cost from supplier i to supermarket j
    demand[j]       : demand of supermarket j

Decision Variables
    y[i]   ∈ {0,1}          (1 if supplier i is activated)
    x[i,j] ≥ 0              (quantity supplied from i to j)

Objective
    minimize  Σ_{i∈I} fixed_cost[i] * y[i]
              + Σ_{i∈I} Σ_{j∈J} transport_cost[i,j] * x[i,j]

Constraints
    (1) Demand satisfaction
        ∀ j∈J :  Σ_{i∈I} x[i,j] = demand[j]

    (2) Supplier activation
        ∀ i∈I, j∈J :  x[i,j] ≤ demand[j] * y[i]

    (3) Non‑negativity
        ∀ i∈I, j∈J :  x[i,j] ≥ 0
```

---

**Data Mapping Rules**

| Parameter / Variable | Source Table | Row‑ID Column | Column(s) | Notes |
|----------------------|--------------|---------------|-----------|-------|
| `fixed_cost[i]` | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Supplier identifier in `Unnamed: 0` |
| `demand[j]` | `file_0_view_0` | `customer` | `demand` | Supermarket identifier in `customer` |
| `transport_cost[i,j]` | `file_2_view_0` | `Unnamed: 0` | `C1`, `C2` | Matrix with rows = suppliers (`Unnamed: 0`) and columns = supermarkets (`customer`) |

All indices `i` and `j` are taken from the full set of entities returned in the observation; no additional filtering is applied. The model uses the exact column names and table identifiers as provided.