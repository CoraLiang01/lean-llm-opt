#### Abstract Mathematical Model

Let:

- $I$ = index set of all products classified under ‘27in’ (from table_id: file_0_view_0, column: Product Name, filtered for ‘27in’)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter; table_id: file_0_view_0, column: Revenue)
    - $d_i$ = deterministic demand for product $i$ (parameter; table_id: file_0_view_0, column: Demand)
    - $s_i$ = initial inventory for product $i$ (parameter; table_id: file_0_view_0, column: Initial Inventory)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable; integer, $x_i \geq 0$)

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

- Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- Index set $I$: All rows in table_id: file_0_view_0 (Salesorders.csv) where Product Name contains ‘27in’
- Parameter $A_i$: Revenue (column: Revenue, table_id: file_0_view_0)
- Parameter $d_i$: Demand (column: Demand, table_id: file_0_view_0)
- Parameter $s_i$: Initial Inventory (column: Initial Inventory, table_id: file_0_view_0)
- Variable $x_i$: Decision variable for each $i \in I$