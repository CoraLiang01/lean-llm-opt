#### Abstract Mathematical Model

**Index Set:**

- $I$ : Set of all dairy products, indexed by $i$.

**Parameters:**

- $A_i$ : Revenue per unit for product $i$.
- $d_i$ : Total demand for product $i$ over the sales horizon.
- $I_i$ : Initial inventory available for product $i$.

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

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

3. **Variable Domain:**
   \[
   x_i \in \mathbb{Z}_{+}, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All records in `DairyGoodsSalesDataset.csv` [`file_0_view_0`], column `Full_Product_Name`.
- **Parameter $A_i$:** `Revenue` column in `DairyGoodsSalesDataset.csv` [`file_0_view_0`].
- **Parameter $d_i$:** `Demand` column in `DairyGoodsSalesDataset.csv` [`file_0_view_0`].
- **Parameter $I_i$:** `Initial Inventory` column in `DairyGoodsSalesDataset.csv` [`file_0_view_0`].