#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all dairy products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column "Revenue").
- $d_i$: Demand for product $i$ (from column "Demand").
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory").

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
3. **Nonnegativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All rows in table_id = "file_0_view_0", column "Full_Product_Name".
- **Parameter $A_i$:** table_id = "file_0_view_0", column "Revenue".
- **Parameter $d_i$:** table_id = "file_0_view_0", column "Demand".
- **Parameter $I_i$:** table_id = "file_0_view_0", column "Initial Inventory".

All data is sourced from table_id = "file_0_view_0" (DairyGoodsSalesDataset.csv), using every row and the specified columns.