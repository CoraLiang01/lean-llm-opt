**Mathematical Model**

```
Sets
    S   : {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5}
    C   : {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6}

Parameters
    demand[c]          for c in C
    supply_capacity[s] for s in S
    cost[s,c]          for s in S, c in C

Variables
    x[s,c] ≥ 0         for s in S, c in C

Objective
    minimize  Z = Σ_{s∈S} Σ_{c∈C} cost[s,c] * x[s,c]

Constraints
    /* Demand satisfaction */
    ∀ c ∈ C :   Σ_{s∈S} x[s,c] = demand[c]

    /* Supply capacity */
    ∀ s ∈ S :   Σ_{c∈C} x[s,c] ≤ supply_capacity[s]
```

---

**Data Mapping Rules**

| Parameter          | Source Table ID | Index Column | Value Column | Notes |
|--------------------|-----------------|--------------|--------------|-------|
| `demand[c]`        | `file_0_view_0` | `Customers`  | `demand`     | One row per customer |
| `supply_capacity[s]` | `file_1_view_0` | `Suppliers`  | `supply_capacity` | One row per supplier |
| `cost[s,c]`        | `file_2_view_0` | `Unnamed: 0` (row) | Column named `CustomerX` | Each row corresponds to a supplier; each column corresponds to a customer |

*All indices (`s`, `c`) are taken directly from the string values in the respective index columns.*