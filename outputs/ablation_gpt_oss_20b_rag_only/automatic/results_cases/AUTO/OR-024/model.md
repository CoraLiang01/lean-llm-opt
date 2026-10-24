**Abstract Optimization Model**

```
SETS
    I   {products} ;

PARAMETERS
    rev[I]   {revenue per unit}
    dem[I]   {demand quantity}
    inv[I]   {initial inventory} ;

VARIABLES
    x[I]     {units of product i to sell} ;

BINARY   x[I]   ;          # integer decision (units are discrete)
POSITIVE x[I]   ;          # x[I] ≥ 0

OBJ: MAX
    sum {i in I} rev[i] * x[i] ;

CONSTRAINTS
    demand_cap {i in I} : x[i] <= dem[i] ;
    inventory_cap {i in I} : x[i] <= inv[i] ;
```

---

**Data Mapping**

| Symbol | Source Table | Column Name |
|--------|--------------|-------------|
| I      | file_0_view_0 | Product Name |
| rev[I] | file_0_view_0 | Revenue |
| dem[I] | file_0_view_0 | Demand |
| inv[I] | file_0_view_0 | Initial Inventory |

*All parameters and the index set are populated from the rows of `file_0_view_0` that satisfy the filter on `Product Name` (prefix `S700_`). No numeric values are hard‑coded; the model remains symbolic and ready for data loading.*