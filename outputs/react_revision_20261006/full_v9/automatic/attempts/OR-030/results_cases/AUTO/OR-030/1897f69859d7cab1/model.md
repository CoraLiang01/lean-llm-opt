#### Symbolic Mathematical Model

Let:
- $I$ = index set of all car models with Product Name prefix ‘FDK57’ (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of car model $i$ (parameter from column ‘Revenue’)
    - $d_i$ = total demand for car model $i$ (parameter from column ‘Demand’)
    - $s_i$ = initial inventory for car model $i$ (parameter from column ‘Initial Inventory’)
    - $x_i$ = number of units of car model $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

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
    - Index set $I$: all rows where [Product Name] has prefix ‘FDK57’
    - Parameter $A_i$: [Revenue]
    - Parameter $d_i$: [Demand]
    - Parameter $s_i$: [Initial Inventory]