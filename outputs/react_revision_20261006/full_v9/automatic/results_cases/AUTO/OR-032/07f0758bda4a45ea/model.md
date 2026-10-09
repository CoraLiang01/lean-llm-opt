#### Symbolic Mathematical Model

Let:
- $\mathcal{I}$ = set of all products classified under ‘Books’ (from Product_Name column, filtered as specified)
- For each $i \in \mathcal{I}$:
    - $A_i$ = revenue per unit of product $i$ (from Revenue)
    - $d_i$ = deterministic demand for product $i$ (from Demand)
    - $I_i$ = initial inventory for product $i$ (from Initial Inventory)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in \mathcal{I} \quad \text{(Demand constraint)} \\
& x_i \leq I_i && \forall i \in \mathcal{I} \quad \text{(Inventory constraint)} \\
& x_i \geq 0 && \forall i \in \mathcal{I} \quad \text{(Non-negativity)} \\
\end{align*}
\]

Variable domain:
- $x_i$ are continuous and non-negative (unless otherwise specified by the query).

#### Data Mapping

- Table: file_0_view_0 (from DifferentStoreSales.csv)
    - Index set $\mathcal{I}$: All rows where Product_Name has prefix "Books"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $I_i$: column "Initial Inventory"
    - Decision variable $x_i$: number of units of product $i$ to fulfill

No additional constraints or bounds are imposed beyond those above.