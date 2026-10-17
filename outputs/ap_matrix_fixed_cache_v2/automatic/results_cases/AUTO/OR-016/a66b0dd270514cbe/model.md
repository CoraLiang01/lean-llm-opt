#### Abstract Mathematical Optimization Model

**Index Sets:**
- $P$: Set of all products, indexed by $i$.

**Parameters:**
- $r_i$: Revenue per unit for product $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: Total demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in P$.

**Objective:**
\[
\max \sum_{i \in P} r_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in P
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in P
   \]

---

**Data Mapping:**

- All parameters ($r_i$, $d_i$, $I_i$) and product identifiers are sourced from table_id: file_0_view_0, columns: ‘Product Name’, ‘Revenue’, ‘Demand’, ‘Initial Inventory’, with no additional filters (full table used).