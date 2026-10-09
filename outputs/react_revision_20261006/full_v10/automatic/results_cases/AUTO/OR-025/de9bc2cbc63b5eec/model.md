Mathematical Optimization Model

Index Sets:
Let $I$ be the set of all products with Product Name starting with "TABLET" in table_id file_0_view_0.

Parameters:
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

Decision Variables:
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Constraints:
For all $i \in I$:
- Inventory constraint: $x_i \leq I_i$
- Demand constraint: $x_i \leq d_i$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \ x_i \geq 0$

Data Mapping:
- Index set $I$: All rows in table_id file_0_view_0 where "Product Name" starts with "TABLET"
- $A_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $I_i$: file_0_view_0, column "Initial Inventory"
- $x_i$: Decision variable for each $i \in I$