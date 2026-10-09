#### Sets
Let $I$ be the set of all products with $\text{Sub Category}$ prefix "Organ" (from the data).

#### Parameters
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id: file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory", table_id: file_0_view_0)

#### Decision Variables
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective
$$
\max \sum_{i \in I} A_i x_i
$$

#### Constraints
For all $i \in I$:
- $x_i \leq d_i$  (Demand constraint)
- $x_i \leq I_i$  (Inventory constraint)
- $x_i \geq 0$  (Non-negativity and integrality)

#### Data Mapping
- $I$: All rows in table_id file_0_view_0 where "Sub Category" has prefix "Organ"
- $A_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $I_i$: file_0_view_0, column "Initial Inventory"