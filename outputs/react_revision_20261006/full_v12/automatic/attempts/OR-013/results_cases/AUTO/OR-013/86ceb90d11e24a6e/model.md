#### Symbolic Mathematical Model

Let:
- $I$ = set of all "4U" products (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = demand for product $i$ (parameter)
    - $s_i$ = initial inventory of product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

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

- Table: OnlineSalesinUSA.csv
    - Index set $I$: all rows where "Product Name" starts with "4U" (table_id: file_0_view_0, column: "Product Name")
    - Parameter $A_i$: "Revenue" column (table_id: file_0_view_0, column: "Revenue")
    - Parameter $d_i$: "Demand" column (table_id: file_0_view_0, column: "Demand")
    - Parameter $s_i$: "Initial Inventory" column (table_id: file_0_view_0, column: "Initial Inventory")