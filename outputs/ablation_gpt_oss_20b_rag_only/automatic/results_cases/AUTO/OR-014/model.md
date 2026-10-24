**MODEL**

```
SETS
    P   : set of product names

PARAMETERS
    Revenue[p]   : revenue per unit of product p
    Demand[p]    : demand for product p
    Inventory[p] : initial inventory of product p

VARIABLES
    x[p]   : integer, units of product p to fulfill (x[p] ≥ 0)

OBJECTIVE
    maximize Z = Σ_{p∈P} Revenue[p] * x[p]

CONSTRAINTS
    x[p] ≤ Demand[p]      ∀p∈P
    x[p] ≤ Inventory[p]   ∀p∈P
```

**DATA MAPPING**

```
P          : table_id "file_0_view_0", column "Product Name"
Revenue[p] : table_id "file_0_view_0", column "Revenue"
Demand[p]  : table_id "file_0_view_0", column "Demand"
Inventory[p]: table_id "file_0_view_0", column "Initial Inventory"
```