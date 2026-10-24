**Abstract Optimization Model**

```
SETS
    I   :  {i | product name contains '27in'}   // 27‑inch products

PARAMETERS
    rev[i]   : revenue per unit of product i
    dem[i]   : demand quantity of product i
    inv[i]   : initial inventory of product i

VARIABLES
    x[i]     : units of product i to fulfill   (continuous,  x[i] ≥ 0)

OBJ
    maximize  Σ_{i∈I} rev[i] * x[i]

CONSTRAINTS
    1.  x[i] ≤ dem[i]          ∀ i∈I          // cannot exceed demand
    2.  x[i] ≤ inv[i]          ∀ i∈I          // cannot exceed available inventory
```

---

**Data Mapping**

| Table ID          | Column Name          | Mapped to                     |
|-------------------|----------------------|--------------------------------|
| file_0_view_0     | Revenue              | `rev[i]`                      |
| file_0_view_0     | Demand               | `dem[i]`                      |
| file_0_view_0     | Initial Inventory    | `inv[i]`                      |
| file_0_view_0     | Product Name         | index set `I` (filter: contains “27in”) |

*All parameters are sourced directly from the specified columns of the filtered view `file_0_view_0`. No numeric values or record counts are included in the model.*