#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with SKU prefix ‘ZZ’ (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter, from column "Demand")
    - $s_i$ = initial inventory of product $i$ (parameter, from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \in \mathbb{Z}_+, && \forall i \in I \quad \text{(Nonnegative integer variables)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (RetailStoreSalesTransactions(ScannerData).csv)
    - Index set $I$: All rows where column "SKU" has prefix "ZZ"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Variable $x_i$: number of units fulfilled for SKU $i$ (decision variable for each $i \in I$)