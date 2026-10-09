#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products classified under ‘Organ’ (as identified by the relevant subcategories in the data).

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total deterministic demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory available for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

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

- **Table:** file_0_view_0 (from SupermartGrocerySales-RetailAnalyticsDataset.csv)
- **Columns Used:**
  - ‘Sub Category’ (to identify products classified under ‘Organ’)
  - ‘Revenue’ (parameter $A_i$)
  - ‘Demand’ (parameter $d_i$)
  - ‘Initial Inventory’ (parameter $I_i$)
- **Selection:** All records in the table are used, as the query fallback returned the full dataset. The set $I$ is defined as all products in the table classified under ‘Organ’ according to the user’s classification scheme.

---

This model maximizes total revenue from fulfilling demand for ‘Organ’ products, subject to inventory and demand limits, using the provided data columns.