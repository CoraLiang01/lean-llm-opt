Mathematical Optimization Model

Index Sets:
Let $I$ be the set of all products with names beginning with "S700_" as defined in table_id file_0_view_0, column "Product Name".

Parameters:
For each $i \in I$:
- $A_i$: revenue per unit of product $i$ (from column "Revenue", table_id file_0_view_0)
- $d_i$: total demand for product $i$ (from column "Demand", table_id file_0_view_0)
- $I_i$: initial inventory of product $i$ (from column "Initial Inventory", table_id file_0_view_0)

Decision Variables:
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill, integer, $x_i \geq 0$

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Constraints:
For all $i \in I$:
$$
x_i \leq d_i
$$
$$
x_i \leq I_i
$$
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

Data Mapping:
- Index set $I$ is defined by all rows in table_id file_0_view_0, column "Product Name", with prefix "S700_".
- Parameter $A_i$ is from table_id file_0_view_0, column "Revenue".
- Parameter $d_i$ is from table_id file_0_view_0, column "Demand".
- Parameter $I_i$ is from table_id file_0_view_0, column "Initial Inventory".