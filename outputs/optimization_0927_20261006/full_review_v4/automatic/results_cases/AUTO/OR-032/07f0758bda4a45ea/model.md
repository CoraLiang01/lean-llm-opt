#### Abstract Mathematical Model

Let:

- $\mathcal{I}$: Index set of all products classified under ‘Books’ (from the dataset).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (parameter).
    - $I_i$: Initial inventory of product $i$ (parameter).
    - $d_i$: Demand for product $i$ (parameter).
    - $x_i$: Number of units of product $i$ to fulfill (decision variable).

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

- Table: file_0_view_0 (from DifferentStoreSales.csv)
- Columns used:
    - Product_Name (filtered by prefix "Books" to select products classified under ‘Books’)
    - Revenue (parameter $A_i$)
    - Initial Inventory (parameter $I_i$)
    - Demand (parameter $d_i$)

All records returned by the query with the filter: Product_Name starts with "Books". No further filtering or aggregation is performed.