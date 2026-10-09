#### Sets
- $P$: Set of all products (indexed by $i$).

#### Parameters
- $A_i$: Revenue per unit for product $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: Demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in P$.

#### Objective
\[
\max \sum_{i \in P} A_i \cdot x_i
\]

#### Constraints
1. Inventory constraints:
   \[
   x_i \leq I_i \quad \forall i \in P
   \]
2. Demand constraints:
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in P
   \]

---

#### Data Mapping

- Source Table: file_0_view_0 (RetailSalesDataset.csv)
- Columns used:
    - Product Name (for index set $P$)
    - Revenue (parameter $A_i$)
    - Demand (parameter $d_i$)
    - Initial Inventory (parameter $I_i$)
- No filters applied; all records from the table are included.