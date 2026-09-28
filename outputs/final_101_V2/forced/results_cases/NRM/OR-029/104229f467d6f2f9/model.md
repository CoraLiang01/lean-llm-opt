#### Abstract Mathematical Optimization Model

Let:

- $I$ = index set of all products classified as ‘FAUX’ (from ZARASales.csv, column "Product Name")
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column "Revenue")
    - $d_i$ = deterministic demand for product $i$ (from column "Demand")
    - $s_i$ = initial inventory for product $i$ (from column "Initial Inventory")
    - $x_i$ = integer decision variable: number of units of product $i$ to fulfill

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

- Table: ZARASales.csv
    - Index set $I$: All rows where "Product Name" contains or starts with "FAUX"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Variable $x_i$: number of units of product $i$ to fulfill

No additional constraints or data sources are imposed by the query.