**Model**

```
Sets
    I   : = {A1,…,A15}          // factories
    J   : = {B1,…,B8}           // distribution centers

Parameters
    FixedCost[i]   : fixed construction cost of factory i
    Capacity[i]    : production capacity of factory i
    ShippingCost[i,j] : unit shipping cost from factory i to center j
    Demand[j]      : required demand at center j

Decision Variables
    y[i] ∈ {0,1}   // 1 if factory i is built
    x[i,j] ≥ 0     // quantity shipped from i to j

Objective
    minimize  Σ_{i∈I} FixedCost[i]·y[i]
              + Σ_{i∈I} Σ_{j∈J} ShippingCost[i,j]·x[i,j]

Constraints
    // Demand satisfaction
    ∀ j∈J :  Σ_{i∈I} x[i,j]  ≥  Demand[j]

    // Capacity limits (only if factory built)
    ∀ i∈I :  Σ_{j∈J} x[i,j]  ≤  Capacity[i]·y[i]
```

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| FixedCost[i] | `file_0_view_0` | `FixedCost` |
| Capacity[i]  | `file_0_view_0` | `Capacity` |
| ShippingCost[i,j] | `file_1_view_0` | `B1`–`B8` (columns indexed by destination) |
| Demand[j] | `file_2_view_0` | `Demand` |

All indices `i` and `j` are taken from the full set of rows returned in the corresponding tables. No additional filtering or enumeration is applied.