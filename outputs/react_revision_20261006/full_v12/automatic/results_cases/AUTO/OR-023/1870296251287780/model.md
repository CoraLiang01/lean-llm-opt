#### Symbolic Mathematical Model

Let $I$ be the set of all products with Product_Reference starting with "ELE-S" in table_id file_0_view_0.

Parameters (for each $i \in I$):
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $s_i$: Initial Inventory of product $i$ (from column "Initial Inventory")

Decision variables:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Demand constraint: $x_i \leq d_i \quad \forall i \in I$
- Inventory constraint: $x_i \leq s_i \quad \forall i \in I$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where column "Product_Reference" has prefix "ELE-S"
- $A_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $s_i$: file_0_view_0, column "Initial Inventory"
- $x_i$: Decision variable for each $i \in I$