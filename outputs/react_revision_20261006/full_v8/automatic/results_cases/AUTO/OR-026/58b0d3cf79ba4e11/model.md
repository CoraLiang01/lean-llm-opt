#### Symbolic Mathematical Model

Let:
- $I$: index set of all products classified under ‘Fashion’ (from Product Name column, filtered as described)
- For each $i \in I$:
    - $A_i$: revenue per unit of product $i$ (parameter, from Revenue)
    - $d_i$: deterministic demand for product $i$ (parameter, from Demand)
    - $s_i$: initial inventory for product $i$ (parameter, from Initial Inventory)
    - $x_i$: integer decision variable, number of units of product $i$ to fulfill

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \in \mathbb{Z}_+, && \forall i \in I \quad \text{(Nonnegativity and integrality)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (SupermarketSales.csv, filtered to ‘Fashion’ products)
    - Index set $I$: all rows where Product Name starts with "Fashion"
    - $A_i$: column "Revenue"
    - $d_i$: column "Demand"
    - $s_i$: column "Initial Inventory"
    - $x_i$: decision variable for each $i \in I$