#### Mathematical Model

Let $I$ be the set of all product categories indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $s_i$: Initial inventory for product $i$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- $I$: All records in table_id = "file_0_view_0", column "Product Name"
- $A_i$: table_id = "file_0_view_0", column "Revenue"
- $d_i$: table_id = "file_0_view_0", column "Demand"
- $s_i$: table_id = "file_0_view_0", column "Initial Inventory"