#### Symbolic Mathematical Model

Let:
- $I$ = set of all baked goods (indexed by $i$)
- $A_i$ = revenue per unit of baked good $i$ (parameter)
- $d_i$ = deterministic demand for baked good $i$ (parameter)
- $s_i$ = initial inventory for baked good $i$ (parameter)
- $x_i$ = quantity of baked good $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- $I$: All unique "Product Name" in table_id: file_0_view_0, column: "Product Name"
- $A_i$: table_id: file_0_view_0, column: "Revenue"
- $d_i$: table_id: file_0_view_0, column: "Demand"
- $s_i$: table_id: file_0_view_0, column: "Initial Inventory"
- $x_i$: Decision variable for each $i \in I$