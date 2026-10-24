**Mathematical Model (Symbolic)**  

```
Sets
    S   : set of Walmart stores   {S1,…,S11}
    D   : set of customer groups  {C1,…,C12}

Parameters
    demand[d]   : daily demand of customer d
    supply[s]   : daily supply capacity of store s
    cost[s,d]   : unit transportation cost from store s to customer d

Decision Variables
    x[s,d]  ≥ 0   : quantity shipped from store s to customer d

Objective
    minimize   Σ_{s∈S} Σ_{d∈D}  cost[s,d] * x[s,d]

Constraints
    1. Supply limits
          Σ_{d∈D} x[s,d]  ≤  supply[s]          ∀ s∈S

    2. Demand satisfaction
          Σ_{s∈S} x[s,d]  ≥  demand[d]          ∀ d∈D
```

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column(s) | Value Column | Notes |
|-----------|--------------|---------------|--------------|-------|
| `demand[d]` | `file_0_view_0` | `customer` | `demand` | `d` is the customer identifier (C1–C12). |
| `supply[s]` | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | `s` is the store identifier (S1–S11). |
| `cost[s,d]` | `file_2_view_0` | `Unnamed: 0` (row) → `s` | `C1`–`C12` (columns) → `d` | Each cell gives the unit cost from store `s` to customer `d`. |

All indices (`S`, `D`) are taken from the complete set of identifiers present in the respective tables; no additional filtering is applied. The model uses only the data supplied in the observation and respects the objective sense and variable domains specified in the query.