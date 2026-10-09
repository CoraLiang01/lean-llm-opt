#### Mathematical Optimization Model

Let:
- $I$ = set of all products with Product_Reference starting with "ELE-S" (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i, \quad \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in I \quad \text{(Non-negativity and integrality)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (SalesStoreoverview.csv)
    - Index set $I$: All rows where Product_Reference starts with "ELE-S"
    - $A_i$: column "Revenue"
    - $d_i$: column "Demand"
    - $s_i$: column "Initial Inventory"
    - $i$: column "Product_Reference" (serves as unique product identifier)