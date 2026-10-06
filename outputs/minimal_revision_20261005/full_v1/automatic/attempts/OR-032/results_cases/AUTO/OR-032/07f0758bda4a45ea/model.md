#### Symbolic Optimization Model

Let $I$ be the set of all products classified under ‘Books’.

Parameters (for each $i \in I$):
- $A_i$: Revenue per unit of product $i$ (from column Revenue)
- $d_i$: Demand for product $i$ (from column Demand)
- $s_i$: Initial inventory for product $i$ (from column Initial Inventory)

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Demand and Inventory Constraints:
  $$
  0 \leq x_i \leq \min\{d_i,\, s_i\}, \quad \forall i \in I
  $$
- Integrality:
  $$
  x_i \in \mathbb{Z}_+, \quad \forall i \in I
  $$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where Product_Name has prefix 'Books'
- Parameter $A_i$: file_0_view_0, column 'Revenue'
- Parameter $d_i$: file_0_view_0, column 'Demand'
- Parameter $s_i$: file_0_view_0, column 'Initial Inventory'