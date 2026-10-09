Mathematical Model

Index Sets:
Let $I$ be the set of all products with "Product Name" starting with "Baby" in table_id file_0_view_0.

Parameters:
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id file_0_view_0)
- $I_i$: Initial Inventory for product $i$ (from column "Initial Inventory", table_id file_0_view_0)

Decision Variables:
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Constraints:
\[
x_i \leq d_i \quad \forall i \in I
\]
\[
x_i \leq I_i \quad \forall i \in I
\]
\[
x_i \geq 0 \text{ and integer} \quad \forall i \in I
\]

Data Mapping:
- Index set $I$ is defined by all rows in table_id file_0_view_0 where "Product Name" starts with "Baby".
- Parameter $A_i$ is mapped from column "Revenue" in table_id file_0_view_0.
- Parameter $d_i$ is mapped from column "Demand" in table_id file_0_view_0.
- Parameter $I_i$ is mapped from column "Initial Inventory" in table_id file_0_view_0.
- Decision variable $x_i$ is defined for each $i \in I$.