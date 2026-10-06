#### Abstract Mathematical Model

**Index Sets:**
- $I$ : Set of all products classified under ‘id999’ (from table_id: file_0_view_0).

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (column: Revenue, table_id: file_0_view_0).
- $d_i$ : Demand for product $i \in I$ during the sales horizon (column: Demand, table_id: file_0_view_0).
- $I_i$ : Initial inventory of product $i \in I$ (column: Initial Inventory, table_id: file_0_view_0).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
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

- All parameters ($A_i$, $d_i$, $I_i$) and index set $I$ are sourced from table_id: file_0_view_0 in OnlineRetailSalesDataset.csv, using columns:
    - id_number (filtered to ‘id999’)
    - Revenue
    - Demand
    - Initial Inventory