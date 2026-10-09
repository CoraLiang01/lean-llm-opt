#### Symbolic Mathematical Model

Let:
- $I$ = set of all products with SKU prefix "ZZ" (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = demand for product $i$ (parameter)
    - $s_i$ = initial inventory of product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \in \mathbb{Z}_+, && \forall i \in I \quad \text{(Nonnegative integer variables)}
\end{align*}
\]

#### Data Mapping

- $I$: All rows in table_id = file_0_view_0, column "SKU" with prefix "ZZ"
- $A_i$: table_id = file_0_view_0, column "Revenue"
- $d_i$: table_id = file_0_view_0, column "Demand"
- $s_i$: table_id = file_0_view_0, column "Initial Inventory"
- $x_i$: Decision variable for each $i \in I$