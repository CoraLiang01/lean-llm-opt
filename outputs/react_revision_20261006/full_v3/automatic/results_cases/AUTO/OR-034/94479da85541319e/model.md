#### Sets
Let $I$ be the set of all baked goods in the bakery, indexed by $i$.

#### Parameters
- $a_i$: Revenue per unit of baked good $i$ (from table_id: file_0_view_0, column: Revenue)
- $d_i$: Demand for baked good $i$ (from table_id: file_0_view_0, column: Demand)
- $s_i$: Initial inventory of baked good $i$ (from table_id: file_0_view_0, column: Initial Inventory)

#### Decision Variables
- $x_i$: Quantity of baked good $i$ to fulfill, for all $i \in I$; $x_i \in \mathbb{Z}_+$

#### Objective
\[
\max \sum_{i \in I} a_i x_i
\]

#### Constraints
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \\
& x_i \leq s_i, && \forall i \in I \\
& x_i \geq 0, && \forall i \in I \\
& x_i \in \mathbb{Z}, && \forall i \in I
\end{align*}
\]

#### Data Mapping
- $I$: All rows in table_id: file_0_view_0, column: Product Name
- $a_i$: table_id: file_0_view_0, column: Revenue
- $d_i$: table_id: file_0_view_0, column: Demand
- $s_i$: table_id: file_0_view_0, column: Initial Inventory