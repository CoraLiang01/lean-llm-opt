#### Symbolic Optimization Model

Let:
- $I$ = index set of all car models classified as ‘FDK57’ (from Product Name column)
- For each $i \in I$:
    - $A_i$ = revenue per unit of car model $i$ (from Revenue column)
    - $d_i$ = demand for car model $i$ (from Demand column)
    - $s_i$ = initial inventory for car model $i$ (from Initial Inventory column)
    - $x_i$ = number of units of car model $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $I$: All rows where Product Name has prefix ‘FDK57’
    - Parameter $A_i$: Revenue (column ‘Revenue’)
    - Parameter $d_i$: Demand (column ‘Demand’)
    - Parameter $s_i$: Initial Inventory (column ‘Initial Inventory’)
    - Variable $x_i$: Quantity fulfilled for each $i \in I$