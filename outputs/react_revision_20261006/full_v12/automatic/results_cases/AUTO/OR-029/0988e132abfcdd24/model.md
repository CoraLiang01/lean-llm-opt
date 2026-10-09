#### Mathematical Optimization Model

Let $I$ be the set of all products in the dataset whose "Product Name" begins with "FAUX".

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column "Revenue")
- $d_i$: Demand for product $i \in I$ (from column "Demand")
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
\[
x_i \leq d_i \qquad \forall i \in I
\]
\[
x_i \leq s_i \qquad \forall i \in I
\]
\[
x_i \geq 0 \text{ and } x_i \in \mathbb{Z} \qquad \forall i \in I
\]

---

#### Data Mapping

- **Index set $I$:** All rows in table_id = file_0_view_0 where "Product Name" starts with "FAUX" (from column "Product Name" in ZARASales.csv)
- **Parameter $A_i$:** "Revenue" column in table_id = file_0_view_0
- **Parameter $d_i$:** "Demand" column in table_id = file_0_view_0
- **Parameter $s_i$:** "Initial Inventory" column in table_id = file_0_view_0
- **Variable $x_i$:** Decision variable for each $i \in I$

All data is sourced from table_id = file_0_view_0, columns: "Product Name", "Revenue", "Demand", "Initial Inventory" in ZARASales.csv.