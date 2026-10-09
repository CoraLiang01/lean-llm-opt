#### Abstract Mathematical Optimization Model

**Index Sets**

- $I$: Set of all products, indexed by $i$.

**Parameters**

- $A_i$: Revenue per unit for product $i$ (from column "Revenue" in table_id: file_0_view_0).
- $d_i$: Expected demand for product $i$ during the sales cycle (from column "Demand" in table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in table_id: file_0_view_0).

**Decision Variables**

- $x_i$: Number of orders fulfilled for product $i$, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective Function**

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints**

1. **Inventory Constraints** (cannot fulfill more than available inventory):
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraints** (cannot fulfill more than realized demand):
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality**:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `MobileSalesDataset.csv`)
- **Index Set:** $I$ corresponds to all unique values in column `"Product Name"`.
- **Parameters:**
    - $A_i$: `"Revenue"`
    - $d_i$: `"Demand"`
    - $I_i$: `"Initial Inventory"`
- **Decision Variables:** $x_i$ defined for each $i \in I$.

No additional constraints or relationships are imposed beyond those specified above.