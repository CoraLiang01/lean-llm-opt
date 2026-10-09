#### Abstract Mathematical Model

**Index Sets:**

- $I$: Set of all pizza types (indexed by $i$), corresponding to all unique values in column `Product Name` of table `file_0_view_0`.

**Parameters:**

- $A_i$: Revenue per unit of pizza type $i$ (`Revenue` column, `file_0_view_0`).
- $d_i$: Total demand for pizza type $i$ over the sales horizon (`Demand` column, `file_0_view_0`).
- $I_i$: Initial inventory available for pizza type $i$ (`Initial Inventory` column, `file_0_view_0`).

**Decision Variables:**

- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective Function:**

\[
\max \quad \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**

1. **Inventory Constraints:**
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraints:**
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `PizzaSalesDataset.csv`)
- **Index Set:** $I$ ← `Product Name`
- **Parameter $A_i$:** `Revenue`
- **Parameter $d_i$:** `Demand`
- **Parameter $I_i$:** `Initial Inventory`