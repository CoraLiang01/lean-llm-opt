**Abstract Optimization Model**

```
SETS
    I   : Aalop product identifiers   {i | product name starts with "Aalop"}

PARAMETERS
    r[i]   : revenue per unit of product i          (from column "Revenue")
    d[i]   : demand of product i                    (from column "Demand")
    inv[i] : initial inventory of product i         (from column "Initial Inventory")

VARIABLES
    x[i]   : units of product i to sell              integer,  x[i] ≥ 0

CONSTRAINTS
    1. x[i] ≤ d[i]          ∀ i ∈ I          (cannot sell more than demand)
    2. x[i] ≤ inv[i]        ∀ i ∈ I          (cannot sell more than available inventory)

OBJECTIVE
    maximize  Σ_{i∈I}  r[i] * x[i]                (total revenue)
```

**Data Mapping**

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| r[i]      | file_0_view_0   | Revenue       |
| d[i]      | file_0_view_0   | Demand        |
| inv[i]    | file_0_view_0   | Initial Inventory |

*All indices `i` are taken from the rows of `file_0_view_0` whose `Product Name` column satisfies the prefix filter “Aalop”.*