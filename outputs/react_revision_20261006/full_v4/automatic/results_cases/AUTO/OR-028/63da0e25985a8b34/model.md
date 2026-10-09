#### Sets
Let $I$ be the set of all products, indexed by $i$.

#### Parameters
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory")

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

#### Objective
\[
\max \sum_{i \in I} A_i x_i
\]

#### Constraints
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \\
& x_i \leq I_i, && \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, && \forall i \in I
\end{align*}
\]

#### Data Mapping
- $I$: All products in table_id = file_0_view_0, column "Product Name"
- $A_i$: table_id = file_0_view_0, column "Revenue"
- $d_i$: table_id = file_0_view_0, column "Demand"
- $I_i$: table_id = file_0_view_0, column "Initial Inventory"