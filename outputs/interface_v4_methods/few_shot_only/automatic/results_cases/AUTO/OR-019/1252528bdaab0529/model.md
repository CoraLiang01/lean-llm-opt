### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products classified as ‘27in’.

#### Parameters
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’ in table_id file_0_view_0).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’ in table_id file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+$ (non-negative integers).

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. Inventory constraints:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraints:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: file_0_view_0 (from SalesDataAnalysis.csv)
    - Product identifier: Product Name
    - Revenue: Revenue
    - Initial Inventory: Initial Inventory
    - Demand: Demand

All parameters are indexed over the set of products classified as ‘27in’ in the source data.