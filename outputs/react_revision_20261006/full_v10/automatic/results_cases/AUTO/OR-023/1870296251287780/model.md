Mathematical Model

Index Sets:
Let $I$ be the set of all products with Product_Reference starting with "ELE-S" as returned from table_id file_0_view_0.

Parameters:
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $s_i$: Initial inventory of product $i$ (from column "Initial Inventory")

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
x_i \leq s_i \quad \forall i \in I
\]
\[
x_i \geq 0 \text{ and } x_i \in \mathbb{Z} \quad \forall i \in I
\]

Data Mapping:
- Index set $I$ is defined by all rows in table_id file_0_view_0 with Product_Reference prefix "ELE-S" from SalesStoreoverview.csv.
- Parameter $A_i$ is mapped from column "Revenue" in table_id file_0_view_0.
- Parameter $d_i$ is mapped from column "Demand" in table_id file_0_view_0.
- Parameter $s_i$ is mapped from column "Initial Inventory" in table_id file_0_view_0.