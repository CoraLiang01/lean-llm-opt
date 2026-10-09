#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all product categories (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit for product $i$.
- $d_i$ : Total deterministic demand for product $i$.
- $I_i$ : Initial inventory available for product $i$.

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**

1. **Inventory Constraints:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]

2. **Demand Constraints:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]

3. **Variable Domain:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `RetailSalesDataset.csv`)
- **Index Set:** $I$ corresponds to all unique values in column `"Product Name"`.
- **Parameter $A_i$:** Value from column `"Revenue"` for product $i$.
- **Parameter $d_i$:** Value from column `"Demand"` for product $i$.
- **Parameter $I_i$:** Value from column `"Initial Inventory"` for product $i$.