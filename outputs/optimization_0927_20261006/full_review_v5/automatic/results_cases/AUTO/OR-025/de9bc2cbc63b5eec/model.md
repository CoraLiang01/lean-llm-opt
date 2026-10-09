#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all ‘TABLET’ smartphone models (indexed by $i$).

**Parameters:**
- $r_i$ : Revenue per unit for model $i$ (from column ‘Revenue’).
- $d_i$ : Total demand for model $i$ (from column ‘Demand’).
- $s_i$ : Initial inventory for model $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : Number of units of model $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} r_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq s_i \quad \forall i \in I
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

- **Table:** SmartphoneRetailOutletSalesData.csv
- **Index Set $I$:** All rows where [Product Name] starts with ‘TABLET_’ (filter: prefix = ‘TABLET_’)
- **Parameter $r_i$:** [Revenue] column, for each $i \in I$
- **Parameter $d_i$:** [Demand] column, for each $i \in I$
- **Parameter $s_i$:** [Initial Inventory] column, for each $i \in I$
- **Model Identifier:** [Product Name] column, for each $i \in I$
- **Table ID:** file_0_view_0

All data is used exactly as returned by the query, with no additional filtering or transformation.