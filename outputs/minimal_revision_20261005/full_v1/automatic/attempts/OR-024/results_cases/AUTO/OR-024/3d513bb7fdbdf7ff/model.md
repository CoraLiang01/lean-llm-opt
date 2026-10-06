#### Symbolic Mathematical Model

Let:
- $I$ = set of all products with identifier prefix ‘S700_’ (from column ‘Product Name’)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column ‘Revenue’)
    - $d_i$ = total demand for product $i$ (from column ‘Demand’)
    - $s_i$ = initial inventory of product $i$ (from column ‘Initial Inventory’)
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
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (SampleSalesData.csv)
    - Index set $I$: All rows where ‘Product Name’ starts with ‘S700_’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’