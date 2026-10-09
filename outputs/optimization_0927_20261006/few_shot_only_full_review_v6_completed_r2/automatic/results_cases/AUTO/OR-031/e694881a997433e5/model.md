---

### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all dairy products, indexed by $i$.

#### Parameters
- $A_i$: Revenue per unit for product $i$, from column `Revenue`.
- $d_i$: Total demand for product $i$, from column `Demand`.
- $I_i$: Initial inventory for product $i$, from column `Initial Inventory`.

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraint** (cannot fulfill more than available inventory):
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint** (cannot fulfill more than demand):
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Nonnegativity and Integrality**:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

### Data Mapping

- **Table**: `file_0_view_0` (from `DairyGoodsSalesDataset.csv`)
- **Index Set**: $I$ is the set of all records in column `Full_Product_Name`.
- **Parameters**:
    - $A_i$: Value from column `Revenue` for product $i$.
    - $d_i$: Value from column `Demand` for product $i$.
    - $I_i$: Value from column `Initial Inventory` for product $i$.
- **Decision Variables**: $x_i$ defined for each $i \in I$.

**Selection**: All records in the table are included; no filtering or aggregation is applied. Each product in the dataset is indexed as $i \in I$.

---