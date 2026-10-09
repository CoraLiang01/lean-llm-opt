#### Abstract Mathematical Optimization Model

Let:

- $I$ = index set of all car model records classified under ‘FDK57’ (from the data).
- For each $i \in I$:
    - $A_i$ = revenue per unit for car model $i$ (parameter, from column ‘Revenue’).
    - $d_i$ = deterministic demand for car model $i$ (parameter, from column ‘Demand’).
    - $s_i$ = initial inventory for car model $i$ (parameter, from column ‘Initial Inventory’).
    - $x_i$ = quantity of car model $i$ to fulfill (decision variable, integer, $x_i \geq 0$).

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i, \quad \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \in \mathbb{Z}_{+}, \quad \forall i \in I \quad \text{(Nonnegativity and integrality)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $I$: all records where ‘Product Name’ = ‘FDK57’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’
    - Decision variable $x_i$: quantity fulfilled for each $i \in I$

No additional constraints or selection logic are imposed beyond those stated above. All data and index sets are defined by the exact filter: ‘Product Name’ = ‘FDK57’ in file_0_view_0.