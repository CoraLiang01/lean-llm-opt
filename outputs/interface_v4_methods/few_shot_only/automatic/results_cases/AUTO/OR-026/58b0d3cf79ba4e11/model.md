#### Abstract Mathematical Model

**Index Sets:**

- $I$: Set of all products in the 'Fashion' category (indexed by $i$).

**Parameters:**

- $A_i$: Revenue per unit of product $i$ (from column 'Revenue', table_id: file_0_view_0).
- $d_i$: Total demand for product $i$ (from column 'Demand', table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column 'Initial Inventory', table_id: file_0_view_0).

**Decision Variables:**

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**

$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**

1. **Inventory Constraint:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraint:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** file_0_view_0 (from SupermarketSales.csv)
- **Columns:**
  - 'Product Name': Used to define index set $I$ (all rows where 'Category' = 'Fashion')
  - 'Revenue': Parameter $A_i$
  - 'Demand': Parameter $d_i$
  - 'Initial Inventory': Parameter $I_i$