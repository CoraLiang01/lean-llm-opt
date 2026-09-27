#### Abstract Mathematical Model

Let:

- $\mathcal{I}$: Index set of all products classified as ‘Books’ (from Product_Name in table_id file_0_view_0).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (parameter from Revenue).
    - $d_i$: Demand for product $i$ (parameter from Demand).
    - $I_i$: Initial inventory for product $i$ (parameter from Initial Inventory).
    - $x_i$: Number of units of product $i$ to fulfill (decision variable).

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
& x_i \geq 0, \quad \forall i \in \mathcal{I} \\
& x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping

- Index set $\mathcal{I}$: All rows in table_id file_0_view_0 where Product_Name has prefix 'Books'.
- Parameter $A_i$: Revenue, column 'Revenue', table_id file_0_view_0.
- Parameter $d_i$: Demand, column 'Demand', table_id file_0_view_0.
- Parameter $I_i$: Initial Inventory, column 'Initial Inventory', table_id file_0_view_0.
- Variable $x_i$: Number of units fulfilled for product $i$ (decision variable for each $i \in \mathcal{I}$).