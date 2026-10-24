**Abstract Optimization Model**

```
SETS
    I   : 27‑inch product identifiers   (filtered by Product Name contains "27in")

PARAMETERS
    rev[i]   : revenue per unit of product i
    dem[i]   : demand quantity of product i
    inv[i]   : initial inventory of product i

VARIABLES
    x[i]     : units of product i to sell   (x[i] ≥ 0)

OBJECTIVE
    maximize  Z = Σ_{i∈I} rev[i] * x[i]

CONSTRAINTS
    1)  x[i] ≤ inv[i]      ∀ i ∈ I          (cannot sell more than inventory)
    2)  x[i] ≤ dem[i]      ∀ i ∈ I          (cannot sell more than demand)
```

---

**Data Mapping**

| Parameter / Set | Source Table ID | Source Column | Notes |
|-----------------|-----------------|---------------|-------|
| `I`             | `file_0_view_0` | `Product Name` | Rows retained after applying filter `Product Name` contains `"27in"` |
| `rev[i]`        | `file_0_view_0` | `Revenue`     | Revenue per unit |
| `dem[i]`        | `file_0_view_0` | `Demand`      | Deterministic demand |
| `inv[i]`        | `file_0_view_0` | `Initial Inventory` | Available inventory |

All parameters are mapped directly from the specified columns of the filtered view `file_0_view_0`. No additional data or numeric values are introduced.