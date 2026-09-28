#### Abstract Mathematical Model

Let:
- $\mathcal{I}$: Index set of all products classified as ‘27in’ (from the source table).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (parameter).
    - $I_i$: Initial inventory of product $i$ (parameter).
    - $d_i$: Demand for product $i$ (parameter).
    - $x_i$: Number of units of product $i$ to fulfill (decision variable).

Objective:
\[
\max \quad \sum_{i \in \mathcal{I}} A_i \cdot x_i
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

- Table: Salesorders.csv (table_id: file_0_view_0)
    - Index set $\mathcal{I}$: All rows where [Product Name] has prefix ‘27in’
    - $A_i$: [Revenue] column
    - $I_i$: [Initial Inventory] column
    - $d_i$: [Demand] column
    - Product identifier: [Product Name] column

No additional constraints or data sources are imposed by the query.