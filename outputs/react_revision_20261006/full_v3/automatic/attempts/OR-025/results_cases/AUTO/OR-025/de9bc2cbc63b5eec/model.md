#### Mathematical Optimization Model

Let $I$ be the set of all products classified under ‘TABLET’ in the dataset.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column "Revenue")
- $d_i$: Demand for product $i \in I$ (from column "Demand")
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
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

#### Data Mapping

- Index set $I$: All records in table_id = "file_0_view_0" where "Product Name" has prefix "TABLET"
- Parameter $A_i$: "Revenue" column, table_id = "file_0_view_0"
- Parameter $d_i$: "Demand" column, table_id = "file_0_view_0"
- Parameter $s_i$: "Initial Inventory" column, table_id = "file_0_view_0"
- Decision variable $x_i$: defined for each $i \in I$