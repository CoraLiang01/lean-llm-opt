#### Sets
- $I$: Index set of all products classified under ‘FAUX’ in the source data.

#### Parameters
- $r_i$: Revenue per unit of product $i \in I$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column "Demand", table_id: file_0_view_0).
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory", table_id: file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $0 \leq x_i \leq \min\{d_i, s_i\}$.

#### Objective
$$
\max \sum_{i \in I} r_i \cdot x_i
$$

#### Constraints
1. Inventory and Demand Fulfillment:
   $$
   0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
   $$
   (Or equivalently, two constraints:)
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$
   $$
   x_i \leq s_i \qquad \forall i \in I
   $$
   $$
   x_i \geq 0, \quad x_i \in \mathbb{Z} \qquad \forall i \in I
   $$

#### Data Mapping

- $I$: All rows in table_id: file_0_view_0 where "Product Name" has prefix "FAUX".
- $r_i$: "Revenue" column, table_id: file_0_view_0.
- $d_i$: "Demand" column, table_id: file_0_view_0.
- $s_i$: "Initial Inventory" column, table_id: file_0_view_0.
- $x_i$: Decision variable for each $i \in I$.

All parameters are mapped directly from the specified columns in table_id: file_0_view_0 (ZARASales.csv).