**Mathematical Model**

```
SETS
    I   : set of product identifiers (Product_Name) that start with “Books”.

PARAMETERS
    Revenue[i]   : revenue per unit of product i.
    Demand[i]    : deterministic demand quantity of product i.
    Inv[i]       : initial inventory level of product i.

VARIABLES
    x[i]         : number of units of product i to be sold.  (integer,  x[i] ≥ 0)

OBJECTIVE
    maximize  Z = Σ_{i∈I} Revenue[i] * x[i]

CONSTRAINTS
    1.  x[i] ≤ Demand[i]          ∀ i ∈ I          (cannot sell more than demand)
    2.  x[i] ≤ Inv[i]             ∀ i ∈ I          (cannot sell more than inventory)
```

---

**Data Mapping**

| Parameter | Table ID          | Column Name          |
|-----------|-------------------|----------------------|
| Revenue   | file_0_view_0     | Revenue              |
| Demand    | file_0_view_0     | Demand               |
| Inv       | file_0_view_0     | Initial Inventory    |
| Index set I | file_0_view_0   | Product_Name (filtered by prefix “Books”) |

*The index set `I` is constructed from the `Product_Name` column of `file_0_view_0` after applying the filter that selects rows whose `Product_Name` starts with “Books”. No other rows are included.*