#### Abstract Optimization Model

**Index Set:**
- $I$ : Set of all products in table_id = file_0_view_0 whose "Product Name" contains the substring "27in".

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (from column "Revenue" in file_0_view_0).
- $d_i$ : Demand for product $i \in I$ (from column "Demand" in file_0_view_0).
- $I_i$ : Initial inventory for product $i \in I$ (from column "Initial Inventory" in file_0_view_0).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint for each $i \in I$:
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
2. Demand constraint for each $i \in I$:
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** file_0_view_0 (from SalesDataAnalysis.csv)
- **Index Set $I$:** All rows where "Product Name" contains "27in"
- **Parameter $A_i$:** Column "Revenue" in file_0_view_0
- **Parameter $d_i$:** Column "Demand" in file_0_view_0
- **Parameter $I_i$:** Column "Initial Inventory" in file_0_view_0

No other columns or rows are used. All data is mapped exactly as described above.