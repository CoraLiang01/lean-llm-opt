#### Mathematical Model

Let $I$ be the set of all baked goods in the bakery.

Parameters:
- $r_i$: Revenue per unit of baked good $i$, from column "Revenue".
- $d_i$: Demand for baked good $i$, from column "Demand".
- $s_i$: Initial inventory for baked good $i$, from column "Initial Inventory".

Decision Variables:
- $x_i$: Quantity of baked good $i$ to fulfill, $\forall i \in I$.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i, && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \geq 0, && \forall i \in I \quad \text{(Non-negativity)}
\end{align*}
\]

#### Data Mapping

- $I$: All "Product Name" entries in table_id: file_0_view_0, column: "Product Name"
- $r_i$: table_id: file_0_view_0, column: "Revenue"
- $d_i$: table_id: file_0_view_0, column: "Demand"
- $s_i$: table_id: file_0_view_0, column: "Initial Inventory"