#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products with "Product Name" starting with "TABLET" in table_id file_0_view_0.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column "Revenue", table_id file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column "Demand", table_id file_0_view_0).
- $I_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory", table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All rows in table_id file_0_view_0 where "Product Name" starts with "TABLET" (column "Product Name", file_0_view_0).
- **Parameter $A_i$:** "Revenue" column, table_id file_0_view_0.
- **Parameter $d_i$:** "Demand" column, table_id file_0_view_0.
- **Parameter $I_i$:** "Initial Inventory" column, table_id file_0_view_0.
- **Variable $x_i$:** Decision variable for each $i \in I$.

All data is sourced from SmartphoneRetailOutletSalesData.csv, table_id file_0_view_0, columns as mapped above.