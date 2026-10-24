**Mathematical Model**

```
SETS
    I   – Organ products (identified by the “Sub Category” field)

PARAMETERS
    Revenue[i]      – unit revenue of product i
    Demand[i]       – deterministic demand of product i
    InitialInv[i]   – initial inventory of product i

VARIABLES
    x[i]  ≥ 0        – units of product i to be sold (integer)

OBJECTIVE
    maximize  Σ_{i∈I} Revenue[i] * x[i]

CONSTRAINTS
    1.  x[i] ≤ Demand[i]          ∀ i∈I          (cannot sell more than demand)
    2.  x[i] ≤ InitialInv[i]      ∀ i∈I          (cannot sell more than inventory)
```

**Data Mapping**

| Symbol | Table ID          | Column Name          |
|--------|-------------------|----------------------|
| I      | file_0_view_0     | Sub Category         |
| Revenue[i] | file_0_view_0 | Revenue              |
| Demand[i]   | file_0_view_0 | Demand               |
| InitialInv[i] | file_0_view_0 | Initial Inventory    |

*All parameters are taken directly from the returned rows of the filtered view `file_0_view_0`.  
The decision variables `x[i]` are integer and non‑negative, reflecting the indivisible nature of product units.*