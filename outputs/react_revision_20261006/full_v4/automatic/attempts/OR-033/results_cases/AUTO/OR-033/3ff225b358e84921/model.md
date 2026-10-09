#### Mathematical Optimization Model

Let:
- $I$ = set of all products classified under ‘Baby’ (from the data mapping below)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$
    - $d_i$ = deterministic demand for product $i$
    - $s_i$ = initial inventory for product $i$
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \\
& x_i \leq s_i && \forall i \in I \\
& x_i \geq 0,\ x_i \in \mathbb{Z} && \forall i \in I
\end{align*}
\]

#### Data Mapping

- $I$: All rows in table_id = file_0_view_0, column "Product Name"
- $A_i$: table_id = file_0_view_0, column "Revenue"
- $d_i$: table_id = file_0_view_0, column "Demand"
- $s_i$: table_id = file_0_view_0, column "Initial Inventory"
- $x_i$: Decision variable for each $i \in I$