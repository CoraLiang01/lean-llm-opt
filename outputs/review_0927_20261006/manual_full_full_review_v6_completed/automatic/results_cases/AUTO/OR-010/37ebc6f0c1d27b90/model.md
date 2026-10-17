#### Index Sets
- $I$: set of all products (indexed by $i$).

#### Parameters
- $A_i$: revenue per unit for product $i$ (from column “Revenue” in table_id: file_0_view_0).
- $d_i$: total deterministic demand for product $i$ during the sales cycle (from column “Demand” in table_id: file_0_view_0).
- $I_i$: initial inventory for product $i$ (from column “Initial Inventory” in table_id: file_0_view_0).

#### Decision Variables
- $x_i$: number of orders fulfilled for product $i$; $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i x_i
\]

#### Constraints
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]
   (Equivalently, two constraints per product:)
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
   \]

#### Data Mapping

- All parameters ($A_i$, $d_i$, $I_i$) are mapped from table_id: file_0_view_0, columns “Revenue”, “Demand”, and “Initial Inventory” in MobileSalesDataset.csv, using all 71 returned records in source order and without additional filtering. The product index set $I$ is defined by the “Product Name” column in the same table.