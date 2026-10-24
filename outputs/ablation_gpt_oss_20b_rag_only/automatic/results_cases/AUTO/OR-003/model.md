**Mathematical Model (Symbolic)**  

```
Sets
    S   : set of suppliers
    C   : set of customers

Parameters
    supply_capacity[s]   : daily supply capacity of supplier s   (s ∈ S)
    demand[c]            : daily demand of customer c           (c ∈ C)
    cost[s,c]            : transportation cost per unit from s to c   (s ∈ S, c ∈ C)

Decision Variables
    x[s,c] ≥ 0           : quantity shipped from supplier s to customer c   (s ∈ S, c ∈ C)

Objective
    minimize   Σ_{s∈S} Σ_{c∈C}  cost[s,c] * x[s,c]

Constraints
    (Supply capacity)   ∀ s∈S :   Σ_{c∈C} x[s,c]  ≤  supply_capacity[s]

    (Demand satisfaction) ∀ c∈C :   Σ_{s∈S} x[s,c]  =  demand[c]
```

---

**Data Mapping Rules**

| Parameter / Variable | Source Table | Key Column(s) | Value Column(s) | Notes |
|----------------------|--------------|---------------|-----------------|-------|
| `S` | `file_1_view_0` | `Unnamed: 0` | – | Supplier identifiers (S1–S10) |
| `C` | `file_0_view_0` | `customer` | – | Customer identifiers (C1–C10) |
| `supply_capacity[s]` | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | Daily capacity per supplier |
| `demand[c]` | `file_0_view_0` | `customer` | `demand` | Daily demand per customer |
| `cost[s,c]` | `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C10` (columns) | Transportation cost matrix |
| `x[s,c]` | – | – | – | Decision variable, non‑negative |

*All indices `s` and `c` are taken from the full set of suppliers and customers present in the respective tables; no additional filtering is applied.*