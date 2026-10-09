Mathematical Model

Index Sets:
Let $I$ be the set of all products whose "Product Name" contains '27in' in table_id file_0_view_0.

Parameters:
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id file_0_view_0)
- $s_i$: Initial Inventory of product $i$ (from column "Initial Inventory", table_id file_0_view_0)

Decision Variables:
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, integer, $0 \leq x_i \leq \min\{d_i, s_i\}$

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Constraints:
For all $i \in I$:
$$
0 \leq x_i \leq d_i
$$
$$
0 \leq x_i \leq s_i
$$
$$
x_i \in \mathbb{Z}
$$

Data Mapping:
- Index set $I$ is defined by all rows in table_id file_0_view_0 (file Salesorders.csv) where "Product Name" contains '27in'.
- Parameter $A_i$ is from column "Revenue" in table_id file_0_view_0.
- Parameter $d_i$ is from column "Demand" in table_id file_0_view_0.
- Parameter $s_i$ is from column "Initial Inventory" in table_id file_0_view_0.
- Decision variable $x_i$ is defined for each $i \in I$ as above.