#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified under ‘ELE-S’ (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

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

#### Data Mapping

- **Index Set $I$:** All rows in table_id file_0_view_0 where column ‘Product_Reference’ has prefix ‘ELE-S’.
- **Parameter $A_i$:** Column ‘Revenue’ in table_id file_0_view_0.
- **Parameter $d_i$:** Column ‘Demand’ in table_id file_0_view_0.
- **Parameter $I_i$:** Column ‘Initial Inventory’ in table_id file_0_view_0.
- **Variable $x_i$:** Decision variable for each $i \in I$.

Data source: SalesStoreoverview.csv, table_id file_0_view_0, filtered to rows where ‘Product_Reference’ has prefix ‘ELE-S’, using columns ‘Revenue’, ‘Initial Inventory’, and ‘Demand’.