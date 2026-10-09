#### Mathematical Optimization Model

Let:
- $I$ = index set of all products classified under ‘Baby’ (from Product Name column, filtered as specified)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from Revenue column)
    - $d_i$ = total demand for product $i$ (parameter from Demand column)
    - $s_i$ = initial inventory of product $i$ (parameter from Initial Inventory column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table ID: file_0_view_0
    - Index set $I$: All rows where Product Name starts with "Baby"
    - $A_i$: Revenue (column "Revenue")
    - $d_i$: Demand (column "Demand")
    - $s_i$: Initial Inventory (column "Initial Inventory")