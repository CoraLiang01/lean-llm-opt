#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified under ‘Fashion’ (from column "Product Name" in table_id file_0_view_0).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from column "Revenue" in table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column "Demand" in table_id file_0_view_0).
- $s_i$: Initial inventory for product $i$ (from column "Initial Inventory" in table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- **Index Set $I$:** All rows in table_id file_0_view_0 where "Product Name" has prefix "Fashion".
- **Parameter $r_i$:** "Revenue" column, table_id file_0_view_0.
- **Parameter $d_i$:** "Demand" column, table_id file_0_view_0.
- **Parameter $s_i$:** "Initial Inventory" column, table_id file_0_view_0.
- **Variable $x_i$:** Decision variable for each $i \in I$.

All data is sourced from table_id file_0_view_0, columns: "Product Name", "Revenue", "Demand", "Initial Inventory".