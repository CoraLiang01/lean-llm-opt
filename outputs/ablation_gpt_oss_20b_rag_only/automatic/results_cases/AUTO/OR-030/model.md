**Abstract Optimization Model**

```text
# Index set – all FDK57 product names
set I;

# Parameters (one value per product)
param revenue{I};
param demand{I};
param inventory{I};

# Decision variables – quantity of each product to sell
var x{I} integer >= 0;

# Objective – maximize total revenue
maximize TotalRevenue:
    sum{i in I} revenue[i] * x[i];

# Constraints – cannot sell more than demand or available inventory
subject to
    DemandLimit{i in I}:
        x[i] <= demand[i];

    InventoryLimit{i in I}:
        x[i] <= inventory[i];
```

**Data Mapping**

| Symbol | Source | Table ID | Column | Notes |
|--------|--------|----------|--------|-------|
| `I` | Index set | `file_0_view_0` | `Product Name` | Filter: `Product Name` starts with prefix `"FDK57"` |
| `revenue[i]` | Parameter | `file_0_view_0` | `Revenue` | |
| `demand[i]` | Parameter | `file_0_view_0` | `Demand` | |
| `inventory[i]` | Parameter | `file_0_view_0` | `Initial Inventory` | |

*All mappings use the exact table ID and column names as provided in the observation.*