**Abstract Optimization Model**

```ampl
# Sets
set I;                     # ZZ‑product SKUs (filtered by prefix “ZZ”)

# Parameters (values come from the CSV view)
param revenue{I};          # revenue per unit
param demand{I};           # deterministic demand
param inventory{I};        # initial inventory

# Decision variables
var x{I} integer >= 0;     # units of product i to sell

# Objective: maximize total revenue
maximize TotalRevenue:
    sum{i in I} revenue[i] * x[i];

# Constraints
subject to
    DemandLimit{i in I}:
        x[i] <= demand[i];          # cannot sell more than demand

    InventoryLimit{i in I}:
        x[i] <= inventory[i];       # cannot sell more than available inventory
```

---

**Data‑Mapping Rules**

| Source | Table ID          | Column Name          | Mapping |
|--------|-------------------|----------------------|---------|
| SKU    | `file_0_view_0`   | `SKU`                | Index set `I` (include only rows where `SKU` starts with `"ZZ"`) |
| Revenue| `file_0_view_0`   | `Revenue`            | Parameter `revenue[i]` |
| Demand | `file_0_view_0`   | `Demand`             | Parameter `demand[i]` |
| Inventory | `file_0_view_0` | `Initial Inventory` | Parameter `inventory[i]` |

*The filter on `SKU` (prefix `"ZZ"`) is applied when constructing the index set `I`.*