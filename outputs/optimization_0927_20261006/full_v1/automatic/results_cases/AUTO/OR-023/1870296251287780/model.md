#### Abstract Mathematical Optimization Model

**Index Set:**

- $I$ : Set of all products classified as ‘ELE-S’ (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective Function:**

\[
\max \quad \sum_{i \in I} A_i \cdot x_i
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

- **Table:** SalesStoreoverview.csv
- **table_id:** file_0_view_0
- **Index Set:** $I$ is all rows where ‘Product_Reference’ starts with ‘ELE-S’
- **Parameters:**
    - $A_i$ : column ‘Revenue’
    - $d_i$ : column ‘Demand’
    - $I_i$ : column ‘Initial Inventory’