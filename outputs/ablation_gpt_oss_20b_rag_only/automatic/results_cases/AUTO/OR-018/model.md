**Abstract Optimization Model**

```
SETS
    I   :  set of Baby products  (derived from rows where Product Name contains "Baby")

PARAMETERS
    revenue[i]   :  revenue per unit of product i   (from column "Revenue")
    demand[i]    :  deterministic demand of product i   (from column "Demand")
    inventory[i] :  initial inventory of product i   (from column "Initial Inventory")

VARIABLES
    x[i]   :  units of product i to sell   (integer, 0 ≤ x[i])

OBJECTIVE
    maximize  Σ_{i∈I}  revenue[i] * x[i]

CONSTRAINTS
    1. 0 ≤ x[i] ≤ inventory[i]          ∀ i ∈ I
    2. 0 ≤ x[i] ≤ demand[i]             ∀ i ∈ I
```

---

**Data Mapping Rules**

| Parameter / Variable | Source Table ID | Source Column |
|----------------------|-----------------|---------------|
| `revenue[i]`         | `file_0_view_0` | `Revenue`     |
| `demand[i]`          | `file_0_view_0` | `Demand`      |
| `inventory[i]`       | `file_0_view_0` | `Initial Inventory` |
| Index set `I`        | `file_0_view_0` | `Product Name` (rows where this column contains the string “Baby”) |

*All mappings use the exact table identifier and column names as provided in the observation.*