**Abstract Optimization Model**

```
MODEL RetailRevenueMaximization

SETS
    I   :  {p | p ∈ ProductName}   // all products in the dataset

PARAMETERS
    revenue[I]   >= 0   // revenue per unit for product i
    demand[I]    >= 0   // maximum demand for product i
    inventory[I] >= 0   // available inventory for product i

VARIABLES
    x[I]   >= 0          // quantity of product i to sell

OBJECTIVE
    maximize Z = sum_{i in I} revenue[i] * x[i]

CONSTRAINTS
    1.  x[i] <= demand[i]      for all i in I
    2.  x[i] <= inventory[i]   for all i in I
```

**Data Mapping Rules**

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| `revenue` | `file_0_view_0` | `"Revenue"` |
| `demand`  | `file_0_view_0` | `"Demand"` |
| `inventory` | `file_0_view_0` | `"Initial Inventory"` |
| Index `I` | `file_0_view_0` | `"Product Name"` |

*The index set `I` is constructed from the distinct values in the `"Product Name"` column of the table `file_0_view_0`.  Each parameter is bound to the corresponding column in that table.*