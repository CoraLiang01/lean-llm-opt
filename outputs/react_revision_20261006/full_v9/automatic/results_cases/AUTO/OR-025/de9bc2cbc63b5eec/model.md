#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all ‘TABLET’ smartphone models (from all records with "Product Name" prefix "TABLET" in the source).

**Parameters:**
- $A_i$: Revenue per unit of model $i \in I$ (from column "Revenue").
- $d_i$: Demand for model $i \in I$ (from column "Demand").
- $I_i$: Initial inventory for model $i \in I$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of model $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. Inventory constraint for each model:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint for each model:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$**: All rows in table_id = file_0_view_0 with "Product Name" prefix "TABLET" (column "Product Name").
- **Parameter $A_i$**: "Revenue" column, table_id = file_0_view_0.
- **Parameter $d_i$**: "Demand" column, table_id = file_0_view_0.
- **Parameter $I_i$**: "Initial Inventory" column, table_id = file_0_view_0.
- **Variable $x_i$**: Decision variable for each $i \in I$.

No additional constraints or bounds are imposed beyond those specified above.