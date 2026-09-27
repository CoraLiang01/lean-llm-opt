#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all ‘TABLET’ smartphone models (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for model $i$ (from column ‘Revenue’).
- $I_i$: Initial inventory for model $i$ (from column ‘Initial Inventory’).
- $d_i$: Demand for model $i$ (from column ‘Demand’).

**Decision Variables:**
- $x_i$: Number of units of model $i$ to fulfill, $\forall i \in I$.

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
3. **Nonnegativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** SmartphoneRetailOutletSalesData.csv
- **Index Set:** All rows where ‘Product Name’ has prefix ‘TABLET’ (table_id: file_0_view_0)
- **Parameters:**
    - $A_i$: ‘Revenue’ column
    - $I_i$: ‘Initial Inventory’ column
    - $d_i$: ‘Demand’ column
- **Decision Variables:** $x_i$ for each $i$ in the above index set

No additional constraints or data sources are imposed beyond those specified above.