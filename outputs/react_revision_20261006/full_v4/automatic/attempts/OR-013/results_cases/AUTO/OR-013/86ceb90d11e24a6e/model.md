##### Sets
Let $\mathcal{I}$ be the set of all "4U" products identified in the data.

##### Parameters
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id: file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory", table_id: file_0_view_0)

##### Decision Variables
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in \mathcal{I}$

##### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i x_i
\]

##### Constraints
\[
x_i \leq d_i, \quad \forall i \in \mathcal{I}
\]
\[
x_i \leq I_i, \quad \forall i \in \mathcal{I}
\]
\[
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I}
\]

##### Data Mapping
- Set $\mathcal{I}$: All records in table_id: file_0_view_0 with "Product Name" prefix "4U"
- Parameter $A_i$: "Revenue" column, table_id: file_0_view_0
- Parameter $d_i$: "Demand" column, table_id: file_0_view_0
- Parameter $I_i$: "Initial Inventory" column, table_id: file_0_view_0