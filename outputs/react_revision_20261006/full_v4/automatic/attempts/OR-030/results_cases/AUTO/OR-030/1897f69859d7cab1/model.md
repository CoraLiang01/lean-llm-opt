#### Mathematical Model

Let $I$ be the set of all car models with Product Name prefix ‘FDK57’ as returned.

Parameters:
- $A_i$: Revenue per unit for car model $i \in I$ (from column ‘Revenue’)
- $d_i$: Demand for car model $i \in I$ (from column ‘Demand’)
- $s_i$: Initial inventory for car model $i \in I$ (from column ‘Initial Inventory’)

Decision variables:
- $x_i$: Number of units of car model $i \in I$ to fulfill (integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \\
& x_i \leq s_i && \forall i \in I \\
& x_i \in \mathbb{Z}_{+} && \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $I$: All rows where ‘Product Name’ has prefix ‘FDK57’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’