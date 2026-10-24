**Model**

```
SETS
    I   : set of products (Product Name)

PARAMETERS
    rev[i]   : revenue per unit of product i
    dem[i]   : demand quantity of product i
    inv[i]   : initial inventory of product i

VARIABLES
    x[i]     : quantity of product i to sell   (continuous,  x[i] ≥ 0)

OBJECTIVE
    maximize Z = Σ_{i∈I} rev[i] * x[i]

CONSTRAINTS
    0 ≤ x[i] ≤ dem[i]   ∀ i∈I
    0 ≤ x[i] ≤ inv[i]   ∀ i∈I
```

**Data Mapping**

| Parameter / Index | Table ID          | Column Name          |
|-------------------|-------------------|----------------------|
| I                 | file_0_view_0     | Product Name         |
| rev[i]            | file_0_view_0     | Revenue              |
| dem[i]            | file_0_view_0     | Demand               |
| inv[i]            | file_0_view_0     | Initial Inventory    |