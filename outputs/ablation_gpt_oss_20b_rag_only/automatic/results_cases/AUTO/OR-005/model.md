**Mathematical model**

```
Sets
  S   : set of suppliers   (suppliers)
  C   : set of customers   (customers)

Parameters
  demand[c]          : daily demand of customer c
  supply_capacity[s] : daily supply capacity of supplier s
  cost[s,c]          : unit transportation cost from supplier s to customer c

Decision variables
  x[s,c] ≥ 0          : quantity shipped from supplier s to customer c

Objective
  minimize  Σ_{s∈S} Σ_{c∈C}  cost[s,c] * x[s,c]

Constraints
  1. Demand satisfaction
        Σ_{s∈S} x[s,c]  =  demand[c]          ∀ c∈C

  2. Supply capacity
        Σ_{c∈C} x[s,c]  ≤  supply_capacity[s]  ∀ s∈S
```

---

**Data‑mapping rules**

| Parameter | Source table | Column(s) | Index mapping |
|-----------|--------------|-----------|---------------|
| `demand[c]` | `file_0_view_0` | `demand` | `Customers` → `c` (exact match) |
| `supply_capacity[s]` | `file_1_view_0` | `supply_capacity` | `Supplier` → `s` (exact match) |
| `cost[s,c]` | `file_2_view_0` | `demand1` … `demand8` | `Unnamed: 0` → `s` via `row_id_mapping` (`supply1`→`supplier1`, …, `supply8`→`supplier8`); column names `demand1`…`demand8` → `c` (exact match) |

*All indices `S` and `C` are taken from the full set of suppliers and customers returned in the observation; no additional filtering is applied.*