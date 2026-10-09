#### Mathematical Optimization Model

**Index Set:**
- $I$ : set of all products with $id\_number$ prefix 'id999' (from data).

**Parameters:**
- $A_i$ : revenue per unit of product $i$ (from column 'Revenue', table_id: file_0_view_0).
- $d_i$ : demand for product $i$ during the sales horizon (from column 'Demand', table_id: file_0_view_0).
- $I_i$ : initial inventory of product $i$ (from column 'Initial Inventory', table_id: file_0_view_0).

**Decision Variables:**
- $x_i$ : number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: OnlineRetailSalesDataset.csv (table_id: file_0_view_0)
    - Index set $I$: all rows where 'id_number' has prefix 'id999'
    - Parameter $A_i$: column 'Revenue'
    - Parameter $d_i$: column 'Demand'
    - Parameter $I_i$: column 'Initial Inventory'