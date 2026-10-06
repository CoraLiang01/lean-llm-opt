### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products classified as ‘FAUX’ (indexed by $i$).

#### Parameters
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: Total demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. Inventory constraint:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Index set $I$ and all parameters ($A_i$, $d_i$, $I_i$) are sourced from table_id: file_0_view_0 in ZARASales.csv, using columns:
    - Product Name (for $i \in I$)
    - Revenue (for $A_i$)
    - Demand (for $d_i$)
    - Initial Inventory (for $I_i$)