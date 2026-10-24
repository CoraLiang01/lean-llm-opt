**Mathematical Model (Symbolic)**  

```
Sets
    W   : set of warehouses
    S   : set of stores

Parameters
    demand[s]          : daily demand of store s          (s ∈ S)
    capacity[w]        : daily supply capacity of warehouse w   (w ∈ W)
    cost[w,s]          : unit transportation cost from w to s   (w ∈ W, s ∈ S)

Decision Variables
    x[w,s] ≥ 0         : quantity shipped from warehouse w to store s   (w ∈ W, s ∈ S)

Objective
    minimize  Σ_{w∈W} Σ_{s∈S}  cost[w,s] * x[w,s]

Constraints
    1. Demand satisfaction (for each store)
          Σ_{w∈W} x[w,s]  =  demand[s]          ∀ s ∈ S

    2. Capacity limits (for each warehouse)
          Σ_{s∈S} x[w,s]  ≤  capacity[w]        ∀ w ∈ W
```

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Mapping Notes |
|-----------|--------------|-----------|---------------|
| `demand[s]` | `file_0_view_0` | `customer`, `demand` | `s` is the value of `customer`; `demand[s]` is the numeric value in column `demand`. |
| `capacity[w]` | `file_1_view_0` | `region`, `supply_capacity` | `w` is the value of `region`; `capacity[w]` is the numeric value in column `supply_capacity`. |
| `cost[w,s]` | `file_2_view_0` | `Unnamed: 0` (warehouse), `D1`–`D5` (stores) | `w` is the value of `Unnamed: 0`; `s` is the column name `D1`–`D5`; `cost[w,s]` is the numeric value in that cell. |

*All indices `W` and `S` are taken from the distinct values appearing in the corresponding columns of the source tables.*