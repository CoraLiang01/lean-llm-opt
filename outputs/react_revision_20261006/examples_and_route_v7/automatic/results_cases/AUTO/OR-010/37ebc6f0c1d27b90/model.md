##### Mathematical Model

Let $I$ be the set of products, indexed by $i$ and identified by the "Product Name" column.

**Parameters:**
- $r_i$: Revenue per unit of product $i$ ("Revenue", table_id: file_0_view_0)
- $d_i$: Demand for product $i$ ("Demand", table_id: file_0_view_0)
- $s_i$: Initial inventory for product $i$ ("Initial Inventory", table_id: file_0_view_0)

**Decision Variables:**
- $x_i$: Number of units of product $i$ fulfilled, $x_i \in \mathbb{Z}_{\geq 0}$

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

##### Data Mapping

- $I$: All products in file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Initial Inventory", keyed by "Product Name"
- $x_i$: Decision variable for each $i \in I$ (fulfilled quantity of product $i$)

**All parameters are mapped directly from the corresponding columns in table_id file_0_view_0.**