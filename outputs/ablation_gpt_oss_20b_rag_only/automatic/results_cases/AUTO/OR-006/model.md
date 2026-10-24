**Symbolic Model**

```
SETS
    W   : {S1,S2,S3,S4,S5,S6,S7,S8,S9,S10}   ! warehouses
    C   : {C1,C2,C3,C4,C5,C6,C7,C8,C9,C10}   ! customers

PARAMETERS
    demand[c]   : daily demand of customer c
    capacity[w] : daily supply capacity of warehouse w
    cost[w,c]   : unit transportation cost from warehouse w to customer c

VARIABLES
    x[w,c]      : quantity shipped from w to c   (continuous,  ≥ 0)

MINIMIZE
    TotalCost = Σ_{w∈W} Σ_{c∈C} cost[w,c] * x[w,c]

CONSTRAINTS
    ! Demand satisfaction
    DemandC[c] : Σ_{w∈W} x[w,c] = demand[c]          ∀ c∈C

    ! Supply capacity
    SupplyW[w] : Σ_{c∈C} x[w,c] ≤ capacity[w]        ∀ w∈W
```

All variables are continuous and non‑negative.  
The model is a linear transportation problem with a single objective of minimizing total cost.

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Mapping Notes |
|-----------|--------------|-----------|---------------|
| `demand[c]` | `file_0_view_0` | `customer` (index), `demand` (value) | Each row gives one customer; `customer` is the index for set `C`. |
| `capacity[w]` | `file_1_view_0` | `Unnamed: 0` (index), `supply_capacity` (value) | Each row gives one warehouse; `Unnamed: 0` is the index for set `W`. |
| `cost[w,c]` | `file_2_view_0` | Matrix: rows indexed by `Unnamed: 0` (warehouses), columns indexed by `C1`…`C10` (customers) | Full 10 × 10 matrix; row and column identifiers align exactly with sets `W` and `C`. |

*All indices are taken directly from the tables; no additional filtering or aggregation is applied.*