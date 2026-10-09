#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$: Set of all pizza types (from column "Product Name" in table_id: file_0_view_0).

**Parameters:**

- $A_i$: Revenue per unit of pizza type $i$ (from column "Revenue" in table_id: file_0_view_0).
- $d_i$: Total demand for pizza type $i$ over the sales horizon (from column "Demand" in table_id: file_0_view_0).
- $I_i$: Initial inventory available for pizza type $i$ (from column "Initial Inventory" in table_id: file_0_view_0).

**Decision Variables:**

- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**

\[
\max \sum_{i \in I} A_i \cdot x_i
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

**Data Mapping:**

- All index sets and parameters ($I$, $A_i$, $d_i$, $I_i$) are defined using columns "Product Name", "Revenue", "Demand", and "Initial Inventory" from table_id: file_0_view_0 in PizzaSalesDataset.csv.
- No additional filters were applied; all rows and columns for all pizza types are included as returned by CSVQA.