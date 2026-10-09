#### Mathematical Optimization Model

Let:
- $I$ = index set of all products with SKU prefix ‘ZZ’ (from the dataset)
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = deterministic demand for product $i$ (parameter)
    - $s_i$ = initial inventory of product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} r_i \cdot x_i
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
    - Parameter $r_i$: Revenue (column: Revenue)
    - Parameter $d_i$: Demand (column: Demand)
    - Parameter $s_i$: Initial Inventory (column: Initial Inventory)