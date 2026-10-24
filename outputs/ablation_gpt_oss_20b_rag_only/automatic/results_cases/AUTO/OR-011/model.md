**Abstract Optimization Model**

```
SET I;                                 // Products with id prefix “id999”

PARAM rev[I]   >= 0;                   // Revenue per unit
PARAM demand[I] >= 0;                  // Known demand during the horizon
PARAM initInv[I] >= 0;                 // Initial inventory available

VAR x[I]  INTEGER >= 0;                // Units of product i to fulfill

MAXIMIZE TotalRevenue:
    sum {i in I} rev[i] * x[i];

CONSTRAINTS
    InventoryLimit {i in I} : x[i] <= initInv[i];
    DemandLimit    {i in I} : x[i] <= demand[i];
```

**Data Mapping**

| Table ID          | Column in source | Model element |
|-------------------|------------------|---------------|
| `file_0_view_0`   | `id_number`      | `I`           |
| `file_0_view_0`   | `Revenue`        | `rev[I]`      |
| `file_0_view_0`   | `Demand`         | `demand[I]`   |
| `file_0_view_0`   | `Initial Inventory` | `initInv[I]` |

*All parameters are bound to the corresponding columns of the returned rows of `file_0_view_0`. No numeric values are included in the model; the formulation remains symbolic and ready for data loading.*