#### Sets

- $I$: Set of all products/categories, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit for product $i$.  
- $d_i$: Demand for product $i$.  
- $s_i$: Initial inventory for product $i$.

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraints:**  
   \[
   x_i \leq s_i \quad \forall i \in I
   \]

2. **Demand Constraints:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]

3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Set $I$**: All records in `file_0_view_0` (table_id: `file_0_view_0`), column `Product Name`.
- **Parameter $A_i$**: `Revenue` column in `file_0_view_0`.
- **Parameter $d_i$**: `Demand` column in `file_0_view_0`.
- **Parameter $s_i$**: `Initial Inventory` column in `file_0_view_0`.