**Abstract Optimization Model**

```
Sets
    I   : set of FAUX products

Parameters
    rev[i]      : revenue per unit of product i   (from column “Revenue”)
    demand[i]   : deterministic demand of product i   (from column “Demand”)
    inv[i]      : initial inventory of product i   (from column “Initial Inventory”)

Variables
    x[i]        : units of product i to be sold
                  (integer, 0 ≤ x[i] ≤ min(demand[i], inv[i]))

Objective
    maximize  Z = Σ_{i∈I} rev[i] * x[i]

Constraints
    0 ≤ x[i] ≤ demand[i]          ∀ i∈I
    0 ≤ x[i] ≤ inv[i]             ∀ i∈I
```

**Data Mapping**

| Symbol | Source Table ID | Column Name          |
|--------|-----------------|----------------------|
| I      | file_0_view_0   | Product Name         |
| rev[i] | file_0_view_0   | Revenue              |
| demand[i] | file_0_view_0 | Demand               |
| inv[i] | file_0_view_0   | Initial Inventory    |

*All parameters are numeric and non‑negative.  The decision variables `x[i]` are integers.*