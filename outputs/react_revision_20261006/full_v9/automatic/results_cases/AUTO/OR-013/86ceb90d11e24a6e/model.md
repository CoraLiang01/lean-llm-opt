#### Symbolic Mathematical Model

Let:
- $I$ = index set of all “4U” products (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$
    - $d_i$ = demand for product $i$ over the sales horizon
    - $s_i$ = initial inventory of product $i$
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
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

- Table: file_0_view_0 (from OnlineSalesinUSA.csv)
    - Index set $I$: All rows where Product Name starts with "4U"
    - $A_i$: column "Revenue"
    - $d_i$: column "Demand"
    - $s_i$: column "Initial Inventory"