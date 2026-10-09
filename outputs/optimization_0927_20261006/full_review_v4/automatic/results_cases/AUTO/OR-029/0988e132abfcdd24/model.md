#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products with names starting with "FAUX" (from column ‘Product Name’ in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

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

- **Index Set $I$:** All records in table_id file_0_view_0 (ZARASales.csv) where ‘Product Name’ has prefix "FAUX".
- **Parameter $A_i$:** Column ‘Revenue’ in table_id file_0_view_0.
- **Parameter $d_i$:** Column ‘Demand’ in table_id file_0_view_0.
- **Parameter $I_i$:** Column ‘Initial Inventory’ in table_id file_0_view_0.
- **Variable $x_i$:** Defined for each $i \in I$.

No additional constraints or data sources are used. All data and constraints are derived directly from the query and the returned records.