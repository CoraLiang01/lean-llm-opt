#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products and categories (as returned in file_0_view_0, column 'Product Name').

**Parameters:**
- $A_i$: Revenue per unit for product $i \in I$ (from 'Revenue', file_0_view_0).
- $d_i$: Total deterministic demand for product $i \in I$ (from 'Demand', file_0_view_0).
- $I_i$: Initial inventory for product $i \in I$ (from 'Initial Inventory', file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units to fulfill for product $i \in I$; $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** file_0_view_0 (from RetailSalesDataset.csv)
- **Columns:**
  - Product Name: Index set $I$
  - Revenue: Parameter $A_i$
  - Demand: Parameter $d_i$
  - Initial Inventory: Parameter $I_i$
- **Filter:** Only records where 'Product Name' starts with 'Electronics', 'Apparel', or 'Homeware' (as applied in CSVQA).
- **Returned Rows:** All records matching the above filter are included directly as $I$.

---

This model maximizes total revenue by allocating inventory to fulfill demand for each product, subject to inventory and demand limits, using the exact data and filters returned by CSVQA.