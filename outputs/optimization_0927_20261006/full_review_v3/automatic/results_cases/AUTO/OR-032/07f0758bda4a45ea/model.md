#### Abstract Mathematical Model

Let:
- $\mathcal{I}$: Index set of all products classified under ‘Books’ (from Product_Name).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from Revenue).
    - $d_i$: Demand for product $i$ (from Demand).
    - $I_i$: Initial inventory for product $i$ (from Initial Inventory).
    - $x_i$: Decision variable; number of units of product $i$ to fulfill.

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \quad \text{(Demand constraint)} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \quad \text{(Inventory constraint)} \\
& x_i \geq 0, \quad \forall i \in \mathcal{I} \quad \text{(Non-negativity)} \\
& x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I} \quad \text{(Integer variables)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from DifferentStoreSales.csv)
    - Product identifier: Product_Name (filtered to prefix "Books_")
    - Revenue: Revenue
    - Demand: Demand
    - Initial Inventory: Initial Inventory

All records returned by the query predicate Product_Name prefix "Books_" are included as the index set $\mathcal{I}$. No further filtering or aggregation is performed.