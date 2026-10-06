#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$: Set of all products/categories (indexed by $i$).

**Parameters:**

- $A_i$: Revenue per unit for product $i$ (from column 'Revenue', table_id: file_0_view_0).
- $d_i$: Total demand for product $i$ (from column 'Demand', table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column 'Initial Inventory', table_id: file_0_view_0).

**Decision Variables:**

- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:** 
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. **Demand Constraint:** 
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:** 
   $$
   x_i \in \mathbb{Z},\ x_i \geq 0 \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** file_0_view_0 (from RetailSalesDataset.csv)
- **Columns:**
  - Product/category identifier: 'Product Name'
  - Revenue per unit: 'Revenue'
  - Initial inventory: 'Initial Inventory'
  - Demand: 'Demand'

All parameters and index sets are defined symbolically and mapped to their exact source columns and table_id. No literal data values or record counts are included.