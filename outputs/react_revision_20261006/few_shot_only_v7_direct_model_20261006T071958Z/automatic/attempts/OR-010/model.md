#### Mathematical Model

Let $I$ be the set of all products in the dataset.

Parameters:
- $A_i$: Revenue per unit for product $i \in I$ (from column "Revenue")
- $d_i$: Demand for product $i \in I$ (from column "Demand")
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory")

Decision variables:
- $x_i$: Number of orders fulfilled for product $i \in I$, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \\
& x_i \leq s_i, && \forall i \in I \\
& x_i \in \mathbb{Z}_+, && \forall i \in I
\end{align*}
\]

#### Data Mapping

- $I$: All records in table_id = file_0_view_0, column "Product Name"
- $A_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $s_i$: file_0_view_0, column "Initial Inventory"