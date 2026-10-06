#### Symbolic Optimization Model

Let:
- $I$ = index set of all products classified under ‘27in’ (from Product Name, filtered as described in Data Mapping)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from Revenue)
    - $d_i$ = deterministic demand for product $i$ (parameter from Demand)
    - $s_i$ = initial inventory for product $i$ (parameter from Initial Inventory)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory constraints:
$$
x_i \leq s_i \quad \forall i \in I
$$

- Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

- Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in I
$$

#### Data Mapping

- Index set $I$ and parameter values $A_i$, $d_i$, $s_i$ are defined by all rows in table_id: file_0_view_0, with columns:
    - Product Name (for $I$)
    - Revenue (for $A_i$)
    - Demand (for $d_i$)
    - Initial Inventory (for $s_i$)