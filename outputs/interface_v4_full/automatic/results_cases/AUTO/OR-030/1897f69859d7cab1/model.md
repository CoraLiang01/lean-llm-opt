#### Abstract Mathematical Optimization Model

Let:

- $\mathcal{I}$: Index set of all car models classified as ‘FDK57’ (from column ‘Product Name’).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of car model $i$ (from column ‘Revenue’).
    - $I_i$: Initial inventory of car model $i$ (from column ‘Initial Inventory’).
    - $d_i$: Demand for car model $i$ (from column ‘Demand’).
    - $x_i$: Decision variable; number of units of car model $i$ to fulfill.

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \quad \text{(Inventory constraint)} \\
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \quad \text{(Demand constraint)} \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I} \quad \text{(Nonnegativity and integrality)}
\end{align*}
\]

#### Data Mapping

- Table: BigMartSales.csv
    - Index set: $\mathcal{I}$ ← rows where ‘Product Name’ has prefix ‘FDK57’ (column ‘Product Name’)
    - Revenue parameter: $A_i$ ← column ‘Revenue’
    - Initial inventory parameter: $I_i$ ← column ‘Initial Inventory’
    - Demand parameter: $d_i$ ← column ‘Demand’