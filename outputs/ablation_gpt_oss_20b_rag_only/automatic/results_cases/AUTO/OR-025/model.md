**Mathematical Model**

```
SETS
    I   :  { product identifiers }   // Product Name

PARAMETERS
    rev[I]   :  revenue per unit of product i
    inv[I]   :  initial inventory of product i
    dem[I]   :  demand for product i

VARIABLES
    x[I]     :  integer, units of product i to sell   (x[i] ∈ ℤ₊)

OBJ
    maximize  Σ_{i∈I} rev[i] * x[i]

CONSTRAINTS
    0 ≤ x[i] ≤ inv[i]          ∀ i∈I
    0 ≤ x[i] ≤ dem[i]          ∀ i∈I
```

**Data Mapping (CSVQA_DATA)**  

| Parameter | Table ID          | Column Name          |
|-----------|-------------------|----------------------|
| rev       | file_0_view_0     | Revenue              |
| inv       | file_0_view_0     | Initial Inventory    |
| dem       | file_0_view_0     | Demand               |
| I (index) | file_0_view_0     | Product Name         |

*All values are taken directly from the rows returned by the observation; no sorting or re‑indexing is performed.*