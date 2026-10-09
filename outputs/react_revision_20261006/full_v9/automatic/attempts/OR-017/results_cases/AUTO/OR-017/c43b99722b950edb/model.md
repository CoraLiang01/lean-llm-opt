#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with SKU prefix ‘ZZ’ (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$
    - $d_i$ = deterministic demand for product $i$
    - $s_i$ = initial inventory of product $i$
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i, \quad \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in I \quad \text{(Nonnegative integer variables)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (RetailStoreSalesTransactions(ScannerData).csv)
    - Index set $I$: All rows where SKU starts with ‘ZZ’ (column: SKU)
    - Parameter $A_i$: Revenue (column: Revenue)
    - Parameter $d_i$: Demand (column: Demand)
    - Parameter $s_i$: Initial Inventory (column: Initial Inventory)
    - Decision variable $x_i$: Number of units of product $i$ to fulfill (for each $i \in I$)