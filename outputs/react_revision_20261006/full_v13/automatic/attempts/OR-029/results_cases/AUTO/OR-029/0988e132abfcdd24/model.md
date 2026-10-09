#### Symbolic Mathematical Model

**Index Sets:**

- $I$: Set of all products classified under ‘FAUX’ in the source data.

**Parameters:**

- $A_i$: Revenue per unit of product $i \in I$ (from column "Revenue").
- $d_i$: Total demand for product $i \in I$ (from column "Demand").
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory").

**Decision Variables:**

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**

$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**

1. **Inventory Constraint:**
   $$
   x_i \leq s_i \quad \forall i \in I
   $$
2. **Demand Constraint:**
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:** All rows in table_id = file_0_view_0, column "Product Name", where "Product Name" starts with "FAUX".
- **Parameter $A_i$:** table_id = file_0_view_0, column "Revenue".
- **Parameter $d_i$:** table_id = file_0_view_0, column "Demand".
- **Parameter $s_i$:** table_id = file_0_view_0, column "Initial Inventory".
- **Variable $x_i$:** Decision variable for each $i \in I$.

All data is sourced from table_id = file_0_view_0, columns as specified above, in the file "ZARASales.csv".