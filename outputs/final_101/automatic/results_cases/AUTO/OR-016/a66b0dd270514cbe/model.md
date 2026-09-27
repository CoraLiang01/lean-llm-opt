#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$: Set of all products/categories (indexed by $i$).

**Parameters:**

- $A_i$: Revenue per unit for product $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: Total demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

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
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

**Data Mapping:**

- Table: file_0_view_0 (from RetailSalesDataset.csv)
    - Index set $I$: All rows (products/categories) in the table.
    - Parameter $A_i$: Column ‘Revenue’
    - Parameter $d_i$: Column ‘Demand’
    - Parameter $I_i$: Column ‘Initial Inventory’