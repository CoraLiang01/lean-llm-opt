**Mathematical Model (Symbolic)**  

```
SETS
    S   : {S1, S2, …, S18}          ! distribution centers
    C   : {C1, C2, …, C18}          ! customer groups

PARAMETERS
    demand[c]   : demand of customer c
    capacity[s] : supply capacity of center s
    cost[s,c]   : transportation cost per unit from s to c

VARIABLES
    x[s,c]  >= 0   ! quantity shipped from s to c

MINIMIZE
    TotalCost = Σ_{s∈S} Σ_{c∈C} cost[s,c] * x[s,c]

CONSTRAINTS
    ! Demand satisfaction
    DemandC[c] : Σ_{s∈S} x[s,c] = demand[c]          ∀ c∈C

    ! Supply capacity
    SupplyS[s] : Σ_{c∈C} x[s,c] ≤ capacity[s]          ∀ s∈S
```

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| `demand[c]` | `file_0_view_0` | `customer`, `demand` | `c` is the value of `customer` |
| `capacity[s]` | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | `s` is the value of `Unnamed: 0` |
| `cost[s,c]` | `file_2_view_0` | matrix with rows `Unnamed: 0` (s) and columns `C1`–`C18` (c) | 18×18 matrix; row and column identifiers align exactly with `S` and `C` |

All indices (`S`, `C`) are taken directly from the rows/columns of the respective tables; no additional filtering or aggregation is applied. The model uses only the data supplied in the observation and respects the objective sense, variable domains, and constraints specified in the original query.