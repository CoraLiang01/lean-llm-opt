**Mathematical model**

```
Sets
    P   = {I, II, III}                     ! products
    E   = {A1, A2, B1, B2, B3}             ! equipment types
    L   = {(e,p) ∈ E×P | e can process p}  ! allowed (equipment,product) pairs

Parameters
    t[e,p]   : processing time per unit of product p on equipment e
    a[e]     : available operating time of equipment e
    c[e]     : equipment cost per hour of equipment e
    raw[p]   : raw‑material cost per unit of product p
    price[p] : selling price per unit of product p

Decision variables
    x[e,p]   ≥ 0   for all (e,p) ∈ L          ! quantity of product p processed on e

Objective
    maximize  Profit
        = Σ_{(e,p)∈L} (price[p] – raw[p]) · x[e,p]
          – Σ_{e∈E} c[e] · Σ_{p:(e,p)∈L} t[e,p] · x[e,p]

Constraints
    ! Equipment‑time limits
    ∀ e ∈ E :   Σ_{p:(e,p)∈L} t[e,p] · x[e,p]  ≤  a[e]
```

*Allowed pairs `L` (derived from the problem statement)*  

```
L = { (A1,I), (A1,II),
      (A2,I), (A2,II), (A2,III),
      (B1,I), (B1,II),
      (B2,I), (B2,III),
      (B3,I) }
```

All other `(e,p)` combinations are excluded from the decision set, so no
additional “x[e,p] = 0” constraints are needed.

--------------------------------------------------------------------

**Data‑mapping rules (table `file_0_view_0`)**

| Parameter | Source column | Source row (filter) | Notes |
|-----------|---------------|---------------------|-------|
| `a[e]` | “Available Equipment Operating Time” | row where “Equipment / Cost” = e | e ∈ {A1,A2,B1,B2,B3} |
| `c[e]` | “Equipment Cost at Full Load (yuan)” | row where “Equipment / Cost” = e | e ∈ {A1,A2,B1,B2,B3} |
| `t[e,p]` | “Product I”, “Product II”, or “Product III” | row where “Equipment / Cost” = e | p ∈ {I,II,III} |
| `raw[p]` | “Product I”, “Product II”, or “Product III” | row where “Equipment / Cost” = “Raw Material Cost (yuan/unit)” | p ∈ {I,II,III} |
| `price[p]` | “Product I”, “Product II”, or “Product III” | row where “Equipment / Cost” = “Unit Price (yuan/unit)” | p ∈ {I,II,III} |

Only the rows whose “Equipment / Cost” value matches the equipment names
(`A1`, `A2`, `B1`, `B2`, `B3`) or the special rows
(`Raw Material Cost (yuan/unit)`, `Unit Price (yuan/unit)`) are used.
No other rows (e.g., `A3`, `B4`) are included because the query
explicitly restricts the equipment set.