#### Mathematical Model

Let:
- $I$ = set of all products with SKU prefix ‘ZZ’ (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = demand for product $i$ (parameter)
    - $s_i$ = initial inventory of product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i x_i
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

- Table: file_0_view_0 (RetailStoreSalesTransactions(ScannerData).csv)
    - Index set $I$: All rows where SKU has prefix ‘ZZ’ (column: SKU)
    - Parameter $A_i$: Revenue (column: Revenue)
    - Parameter $d_i$: Demand (column: Demand)
    - Parameter $s_i$: Initial Inventory (column: Initial Inventory)