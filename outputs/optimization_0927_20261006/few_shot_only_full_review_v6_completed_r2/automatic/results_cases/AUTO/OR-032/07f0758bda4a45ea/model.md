---

### Abstract Mathematical Model

#### Index Sets
- $I$ : Set of all products $i$ such that Product\_Name contains "Books" in table file\_0\_view\_0.

#### Parameters
- $A_i$ : Revenue per unit of product $i$, from column "Revenue" in file\_0\_view\_0.
- $d_i$ : Demand for product $i$, from column "Demand" in file\_0\_view\_0.
- $I_i$ : Initial inventory for product $i$, from column "Initial Inventory" in file\_0\_view\_0.

#### Decision Variables
- $x_i$ : Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory and Demand Bounds:**
   \[
   0 \leq x_i \leq \min\{d_i, I_i\} \quad \forall i \in I
   \]
   (Or equivalently, two separate constraints:)
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

2. **Variable Domain:**
   \[
   x_i \in \mathbb{Z} \quad \forall i \in I
   \]

---

### Data Mapping

- **Source Table:** file\_0\_view\_0 (DifferentStoreSales.csv)
- **Selection:** $I = \{i : \text{Product\_Name contains "Books"}\}$
- **Parameters:**
  - $A_i$ from column "Revenue"
  - $d_i$ from column "Demand"
  - $I_i$ from column "Initial Inventory"
- **Decision Variables:** $x_i$ for each $i \in I$

---