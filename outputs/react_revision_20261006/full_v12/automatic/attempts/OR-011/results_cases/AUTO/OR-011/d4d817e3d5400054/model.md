#### Symbolic Mathematical Model

Let $I$ be the set of all products with $\text{id\_number}$ prefix "id999" from table_id file_0_view_0.

Parameters (for each $i \in I$):
- $A_i$: revenue per unit of product $i$ (from column "Revenue")
- $d_i$: demand for product $i$ during the sales horizon (from column "Demand")
- $I_i$: initial inventory of product $i$ (from column "Initial Inventory")

Decision variables:
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Inventory constraints: $x_i \leq I_i, \quad \forall i \in I$
- Demand constraints: $x_i \leq d_i, \quad \forall i \in I$
- Non-negativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where "id_number" has prefix "id999"
- Parameter $A_i$: column "Revenue" in table_id file_0_view_0
- Parameter $d_i$: column "Demand" in table_id file_0_view_0
- Parameter $I_i$: column "Initial Inventory" in table_id file_0_view_0
- Variable $x_i$: defined for each $i \in I$ as above

All data is sourced from table_id file_0_view_0, columns ["id_number", "Revenue", "Demand", "Initial Inventory"].