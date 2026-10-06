#### Symbolic Optimization Model

Let:
- $I$ = set of all products classified under ‘ZZ’, indexed by $i$
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ (parameter, from column ‘Revenue’)
    - $d_i$ = total demand for product $i$ (parameter, from column ‘Demand’)
    - $s_i$ = initial inventory for product $i$ (parameter, from column ‘Initial Inventory’)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \geq 0,\ x_i \in \mathbb{Z} && \forall i \in I \quad \text{(Nonnegativity and integrality)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from RetailStoreSalesTransactions(ScannerData).csv)
    - Index set $I$: All rows where column ‘SKU’ contains ‘ZZ’
    - Parameter $r_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’