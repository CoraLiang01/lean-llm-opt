**Abstract Mathematical Model**

**Index Set:**
- $I$: set of pizza types (from all rows in PizzaSalesDataset.csv, column "Product Name")

**Parameters:**
- $r_i$: revenue per unit of pizza type $i$ (PizzaSalesDataset.csv, "Revenue")
- $d_i$: demand for pizza type $i$ (PizzaSalesDataset.csv, "Demand")
- $s_i$: initial inventory for pizza type $i$ (PizzaSalesDataset.csv, "Initial Inventory")

**Decision Variables:**
- $x_i$: number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

---

**Data Mapping**

- $I$: All records in PizzaSalesDataset.csv, column "Product Name", table_id = file_0_view_0
- $r_i$: PizzaSalesDataset.csv, column "Revenue", table_id = file_0_view_0, key = "Product Name"
- $d_i$: PizzaSalesDataset.csv, column "Demand", table_id = file_0_view_0, key = "Product Name"
- $s_i$: PizzaSalesDataset.csv, column "Initial Inventory", table_id = file_0_view_0, key = "Product Name"