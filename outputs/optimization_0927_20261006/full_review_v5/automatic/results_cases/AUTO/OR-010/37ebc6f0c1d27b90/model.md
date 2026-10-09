#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit for product $i$ (from column "Revenue").
- $d_i$ : Total deterministic demand for product $i$ over the sales cycle (from column "Demand").
- $I_i$ : Initial inventory for product $i$ (from column "Initial Inventory").

**Decision Variables:**

- $x_i$ : Number of orders fulfilled for product $i$; $x_i \in \mathbb{Z}_+, \forall i \in I$.

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

#### Data Mapping

- **Table:** `file_0_view_0` (from `MobileSalesDataset.csv`)
- **Columns:**
  - Revenue $\rightarrow$ parameter $A_i$
  - Demand $\rightarrow$ parameter $d_i$
  - Initial Inventory $\rightarrow$ parameter $I_i$
  - Product Name $\rightarrow$ index set $I$ (product identifiers)