**Mathematical Model (Facility‑Location / Capacitated‑FLP)**  

```
Sets
    I   : suppliers   (indices of rows in file_1_view_0)
    J   : customers   (indices of rows in file_0_view_0)

Parameters
    fixed_cost[i]   : fixed opening cost of supplier i
    demand[j]       : demand of customer j
    trans_cost[i,j] : transportation cost per unit from supplier i to customer j
    M[i]            : big‑M for supplier i  (e.g. M[i] = Σ_j demand[j])

Decision Variables
    y[i]   ∈ {0,1}          : 1 if supplier i is opened
    x[i,j] ≥ 0              : quantity shipped from i to j

Objective
    minimize  Σ_i fixed_cost[i] * y[i]
              + Σ_i Σ_j trans_cost[i,j] * x[i,j]

Constraints
    1. Demand satisfaction
          ∀ j ∈ J :  Σ_i x[i,j] = demand[j]

    2. Supplier capacity / opening link
          ∀ i ∈ I, ∀ j ∈ J :  x[i,j] ≤ M[i] * y[i]
```

---

**Data‑Mapping Rules**

| Parameter / Variable | Source Table | Key Column(s) | Value Column(s) | Notes |
|----------------------|--------------|---------------|-----------------|-------|
| `I` | `file_1_view_0` | `Unnamed: 0` | – | Supplier identifiers |
| `J` | `file_0_view_0` | `customer` | – | Customer identifiers |
| `fixed_cost[i]` | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Fixed cost per supplier |
| `demand[j]` | `file_0_view_0` | `customer` | `demand` | Demand per customer |
| `trans_cost[i,j]` | `file_2_view_0` | Row: `Unnamed: 0` (supplier) <br>Column: `customer` (customer) | Cell value | Transportation cost matrix |
| `M[i]` | Derived | – | – | `M[i] = Σ_j demand[j]` (constant, can be pre‑computed) |

All indices `i` and `j` are taken from the complete set of suppliers and customers present in the respective CSV files; no additional filtering is applied unless explicitly requested. The model uses a single big‑M per supplier; any tighter bound (e.g., the supplier’s own capacity if available) may replace `M[i]` without changing the formulation.