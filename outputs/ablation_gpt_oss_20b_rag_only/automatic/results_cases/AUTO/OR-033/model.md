**Abstract Optimization Model**

```
SETS
    I   : set of Baby products (derived from rows where "Product Name" contains "Baby")

PARAMETERS
    rev[i]   : revenue per unit of product i   (>= 0)
    dem[i]   : demand quantity of product i   (>= 0)
    inv[i]   : initial inventory of product i (>= 0)

VARIABLES
    x[i]     : units of product i to be fulfilled   (>= 0)

OBJECTIVE
    maximize  Z = sum{i in I} rev[i] * x[i]

CONSTRAINTS
    1.  x[i] <= dem[i]          for all i in I
    2.  x[i] <= inv[i]          for all i in I
```

---

**Data Mapping**

| Source | Table ID          | Column Name          | Mapped to          |
|--------|-------------------|----------------------|--------------------|
| 1      | file_0_view_0     | Product Name         | Index set I (filter: contains “Baby”) |
| 1      | file_0_view_0     | Revenue              | rev[i]             |
| 1      | file_0_view_0     | Demand               | dem[i]             |
| 1      | file_0_view_0     | Initial Inventory    | inv[i]             |

*All parameters and variables are defined symbolically; no numerical values are included.*