**Mathematical model (FLP)**  

```
Sets
    I   : set of plants  (F1,…,F15)
    J   : set of customers (C1,…,C15)

Parameters
    fixed_cost[i]   : fixed opening cost of plant i
    capacity[i]     : maximum supply capacity of plant i
    transport_cost[i,j] : per‑unit transport cost from plant i to customer j
    demand[j]       : demand of customer j

Decision variables
    y[i]   ∈ {0,1}          : 1 if plant i is opened, 0 otherwise
    x[i,j] ≥ 0              : units shipped from plant i to customer j

Objective
    minimize  Σ_{i∈I} fixed_cost[i] · y[i]
              + Σ_{i∈I} Σ_{j∈J} transport_cost[i,j] · x[i,j]

Constraints
    /* Demand satisfaction */
    ∀ j∈J :  Σ_{i∈I} x[i,j]  =  demand[j]

    /* Capacity of opened plants */
    ∀ i∈I :  Σ_{j∈J} x[i,j]  ≤  capacity[i] · y[i]
```

---

**Data‑mapping rules**

| Parameter | Source table | Column | Mapping expression |
|-----------|--------------|--------|---------------------|
| `fixed_cost[i]` | `file_0_view_0` | `fixed_cost` | `fixed_cost[i] = value where plant = i` |
| `capacity[i]` | `file_0_view_0` | `capacity` | `capacity[i] = value where plant = i` |
| `transport_cost[i,j]` | `file_0_view_0` | `Cj` (e.g., `C1`, `C2`, … `C15`) | `transport_cost[i,j] = value where plant = i` |
| `demand[j]` | `file_1_view_0` | `demand` | `demand[j] = value where customer = j` |

*Sets*  
- `I` is the set of distinct `plant` values in `file_0_view_0`.  
- `J` is the set of distinct `customer` values in `file_1_view_0`.

All parameters are taken directly from the corresponding CSV rows; no numerical values are enumerated in the model.