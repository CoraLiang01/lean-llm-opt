#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products classified under ‘Organ’ (from Sub Category column)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from Revenue)
    - $d_i$ = demand for product $i$ (parameter, from Demand)
    - $s_i$ = initial inventory of product $i$ (parameter, from Initial Inventory)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Inventory constraints: $x_i \leq s_i \quad \forall i \in I$
- Demand constraints: $x_i \leq d_i \quad \forall i \in I$
- Variable domain: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

#### Data Mapping

- Table ID: file_0_view_0
    - Index set $I$: All rows where Sub Category contains ‘Organ’
    - Parameter $A_i$: Revenue (column)
    - Parameter $d_i$: Demand (column)
    - Parameter $s_i$: Initial Inventory (column)