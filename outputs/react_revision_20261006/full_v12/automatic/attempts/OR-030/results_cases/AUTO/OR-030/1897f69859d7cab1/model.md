#### Symbolic Mathematical Model

Let $I$ be the index set of all car models with "Product Name" prefix 'FDK57' in table_id file_0_view_0.

Parameters (for each $i \in I$):
- $A_i$: Revenue per unit of car model $i$ (from column "Revenue")
- $d_i$: Demand for car model $i$ (from column "Demand")
- $s_i$: Initial Inventory for car model $i$ (from column "Initial Inventory")

Decision variables:
- $x_i$: Number of units of car model $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Demand and inventory limits:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
$$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 with "Product Name" prefix 'FDK57' (column "Product Name", file_0_view_0)
- $A_i$: "Revenue" column, table_id file_0_view_0
- $d_i$: "Demand" column, table_id file_0_view_0
- $s_i$: "Initial Inventory" column, table_id file_0_view_0
- $x_i$: Decision variable for each $i \in I$