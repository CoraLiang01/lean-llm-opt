#### Symbolic Optimization Model

Let:
- $I$ = index set of all products classified under ‘FAUX’ (from Product Name in table_id: file_0_view_0)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from Revenue)
    - $d_i$ = demand for product $i$ (from Demand)
    - $s_i$ = initial inventory for product $i$ (from Initial Inventory)
    - $x_i$ = integer number of units of product $i$ to fulfill

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory and demand bounds:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I
$$
- Integer variables:
$$
x_i \in \mathbb{Z} \qquad \forall i \in I
$$

#### Data Mapping

- Index set $I$ and all parameters $A_i$, $d_i$, $s_i$ are defined by all rows in table_id: file_0_view_0 of ZARASales.csv, filtered where Product Name contains or starts with ‘FAUX’.
    - $A_i$: column ‘Revenue’
    - $d_i$: column ‘Demand’
    - $s_i$: column ‘Initial Inventory’