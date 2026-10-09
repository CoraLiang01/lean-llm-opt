#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all baked goods, indexed by $i$ (from all "Product Name" entries in table_id file_0_view_0).

**Parameters:**
- $a_i$: Revenue per unit of baked good $i$ (from column "Revenue", table_id file_0_view_0).
- $d_i$: Demand for baked good $i$ (from column "Demand", table_id file_0_view_0).
- $s_i$: Initial inventory of baked good $i$ (from column "Initial Inventory", table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill, for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
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
3. Nonnegativity and integrality:
   \[
   x_i \geq 0,\quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$**: All "Product Name" values from table_id file_0_view_0, column "Product Name".
- **Parameter $a_i$**: table_id file_0_view_0, column "Revenue".
- **Parameter $d_i$**: table_id file_0_view_0, column "Demand".
- **Parameter $s_i$**: table_id file_0_view_0, column "Initial Inventory".