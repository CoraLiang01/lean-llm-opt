#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products classified as ‘Organ’ (as identified in the data source).

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total deterministic demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

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

- **Index Set $I$:** All rows in table_id: file_0_view_0 (SupermartGrocerySales-RetailAnalyticsDataset.csv) where the product is classified as ‘Organ’ (as defined by the user; in the returned data, this would be the subset of rows with ‘Organ’ in the relevant classification column).
- **Parameter $A_i$:** file_0_view_0, column ‘Revenue’, for each $i \in I$.
- **Parameter $d_i$:** file_0_view_0, column ‘Demand’, for each $i \in I$.
- **Parameter $I_i$:** file_0_view_0, column ‘Initial Inventory’, for each $i \in I$.
- **Product Identifier:** file_0_view_0, column ‘Sub Category’ (or other identifier as present in the data).

All data is sourced from table_id: file_0_view_0, columns: ‘Sub Category’, ‘Revenue’, ‘Demand’, ‘Initial Inventory’. The subset $I$ is determined by the ‘Organ’ classification as specified in the user query.