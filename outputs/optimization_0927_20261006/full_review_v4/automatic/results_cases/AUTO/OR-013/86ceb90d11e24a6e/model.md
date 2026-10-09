### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all “4U” products (from column “Product Name” in table_id file_0_view_0).

#### Parameters
- $A_i$: Revenue per unit of product $i$ (from column “Revenue” in table_id file_0_view_0).
- $d_i$: Total demand for product $i$ over the sales horizon (from column “Demand” in table_id file_0_view_0).
- $I_i$: Initial inventory of product $i$ (from column “Initial Inventory” in table_id file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]

#### Data Mapping
- Source table: OnlineSalesinUSA.csv (table_id: file_0_view_0)
- Columns used:
    - “Product Name” (index set $I$)
    - “Revenue” (parameter $A_i$)
    - “Demand” (parameter $d_i$)
    - “Initial Inventory” (parameter $I_i$)
- Filter: Only rows where “Product Name” has prefix “4U” (as returned by CSVQA).