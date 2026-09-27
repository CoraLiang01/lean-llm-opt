#### Abstract Mathematical Model

Let:
- $I$ = index set of all car models classified under ‘FDK57’ (from column ‘Product Name’ with value ‘FDK57’)
- For each $i \in I$:
    - $A_i$ = revenue per unit of car model $i$ (from column ‘Revenue’)
    - $d_i$ = total demand for car model $i$ (from column ‘Demand’)
    - $s_i$ = initial inventory of car model $i$ (from column ‘Initial Inventory’)
    - $x_i$ = integer decision variable: quantity of car model $i$ to fulfill

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

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $I$: All rows where ‘Product Name’ = ‘FDK57’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’