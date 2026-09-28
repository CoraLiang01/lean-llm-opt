#### Abstract Mathematical Optimization Model

Let:

- $\mathcal{I}$: Index set of all car models classified as ‘FDK57’ (from table_id: file_0_view_0, column: Product Name).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of car model $i$ (from column: Revenue).
    - $I_i$: Initial inventory of car model $i$ (from column: Initial Inventory).
    - $d_i$: Demand for car model $i$ (from column: Demand).
    - $x_i$: Decision variable; number of units of car model $i$ to fulfill.

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
& x_i \geq 0, \quad \forall i \in \mathcal{I} \\
& x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $\mathcal{I}$: All rows where Product Name has prefix ‘FDK57’
    - Parameter $A_i$: Revenue (column: Revenue)
    - Parameter $I_i$: Initial Inventory (column: Initial Inventory)
    - Parameter $d_i$: Demand (column: Demand)
    - Variable $x_i$: Fulfilled quantity for car model $i$