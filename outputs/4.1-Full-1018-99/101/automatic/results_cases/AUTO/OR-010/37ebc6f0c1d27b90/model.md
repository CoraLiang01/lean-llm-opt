#### Index Sets
- $I$: set of all products, indexed by $i$.

#### Parameters
- $A_i$: revenue per unit for product $i$ (from column “Revenue” in table_id: file_0_view_0).
- $d_i$: expected demand for product $i$ during the sales cycle (from column “Demand” in table_id: file_0_view_0).
- $I_i$: initial inventory for product $i$ (from column “Initial Inventory” in table_id: file_0_view_0).

#### Decision Variables
- $x_i$: number of orders fulfilled for product $i$, $x_i \in \mathbb{Z}_+$ (non-negative integer), for all $i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]
   (Equivalently, two constraints:)
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

- All parameters are mapped from table_id: file_0_view_0 (file: MobileSalesDataset.csv), using columns:
    - “Product Name” (product identifier, index set $I$)
    - “Revenue” (parameter $A_i$)
    - “Demand” (parameter $d_i$)
    - “Initial Inventory” (parameter $I_i$)
- No filters were applied; all records in source order are included.