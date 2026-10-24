**Abstract Optimization Model**

```
SETS
    P   :  set of products

PARAMETERS
    rev[p]   : revenue per unit of product p
    dem[p]   : demand quantity of product p
    inv[p]   : initial inventory of product p

VARIABLES
    x[p]     : integer units of product p to be fulfilled   (x[p] ∈ ℤ₊)

OBJECTIVE
    maximize  Σ_{p∈P} rev[p] * x[p]

CONSTRAINTS
    1. 0 ≤ x[p] ≤ dem[p]          ∀ p ∈ P          (cannot sell more than demand)
    2. 0 ≤ x[p] ≤ inv[p]          ∀ p ∈ P          (cannot sell more than inventory)
```

---

**Data Mapping Rules**

| Index / Parameter | Source Table ID | Source Column | Mapping Description |
|-------------------|-----------------|---------------|----------------------|
| `P`               | `file_0_view_0` | `Product Name` | Each distinct product name becomes an element of set `P`. |
| `rev[p]`          | `file_0_view_0` | `Revenue`      | For each product `p`, `rev[p]` is the numeric value in the `Revenue` column. |
| `dem[p]`          | `file_0_view_0` | `Demand`       | For each product `p`, `dem[p]` is the numeric value in the `Demand` column. |
| `inv[p]`          | `file_0_view_0` | `Initial Inventory` | For each product `p`, `inv[p]` is the numeric value in the `Initial Inventory` column. |

*All numeric values are parsed as real numbers; decision variables `x[p]` are constrained to be non‑negative integers.*