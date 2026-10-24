**Mathematical Model (Facility‑Location with Capacity)**  

```
SETS
    W   :  {W1,…,W10}          // warehouses
    C   :  {C1,…,C20}          // customers

PARAMETERS
    FixedCost[w]   :  fixed annual opening cost of warehouse w
    Capacity[w]    :  maximum service capacity of warehouse w
    Demand[c]      :  demand of customer c
    Cost[w,c]      :  unit transportation cost from warehouse w to customer c

VARIABLES
    y[w]   ∈ {0,1}              // 1 if warehouse w is opened
    x[w,c] ≥ 0                  // quantity shipped from w to c

MINIMIZE
    Z =  Σ_{w∈W}  FixedCost[w] * y[w]
        + Σ_{w∈W} Σ_{c∈C}  Cost[w,c] * x[w,c]

CONSTRAINTS
    // Demand satisfaction
    (D1)  ∀c∈C :  Σ_{w∈W} x[w,c]  =  Demand[c]

    // Capacity limits (only if warehouse is opened)
    (C1)  ∀w∈W :  Σ_{c∈C} x[w,c]  ≤  Capacity[w] * y[w]

    // Non‑negativity (implicit in variable definition)
    (N1)  ∀w∈W, ∀c∈C :  x[w,c] ≥ 0
```

---

### Data Mapping

| Parameter | Source Table | Column(s) | Mapping Notes |
|-----------|--------------|-----------|---------------|
| `FixedCost[w]` | `file_1_view_0` | `Fixed_Cost` | One‑to‑one mapping on `Warehouse ID` |
| `Capacity[w]`  | `file_1_view_0` | `Capacity`  | One‑to‑one mapping on `Warehouse ID` |
| `Demand[c]`    | `file_2_view_0` | `Demand`    | One‑to‑one mapping on `Customer ID` |
| `Cost[w,c]`    | `file_0_view_0` | `C1`–`C20`  | Matrix mapping: row axis `Warehouse ID`, column axis `Customer ID` |

*All indices (`w`, `c`) are taken from the full set of rows/columns present in the corresponding tables; no additional filtering is applied.*