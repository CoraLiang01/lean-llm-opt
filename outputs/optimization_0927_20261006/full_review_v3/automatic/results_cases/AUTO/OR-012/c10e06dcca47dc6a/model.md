#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$.  
  (From column 'Revenue', table_id: file_0_view_0)
- $d_i$: Total deterministic demand for product $i$ over the sales horizon.  
  (From column 'Demand', table_id: file_0_view_0)
- $I_i$: Initial inventory available for product $i$.  
  (From column 'Initial Inventory', table_id: file_0_view_0)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.  
  ($x_i \in \mathbb{Z}_+, \forall i \in I$)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraints:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraints:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- All parameters ($A_i$, $d_i$, $I_i$) are mapped from columns 'Revenue', 'Demand', and 'Initial Inventory' in table_id: file_0_view_0 (file: OnlineSalesDataset.csv), for all products (no filter applied).