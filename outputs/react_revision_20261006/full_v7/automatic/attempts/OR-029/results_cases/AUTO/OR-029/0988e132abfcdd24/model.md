#### Mathematical Optimization Model

Let:
- $\mathcal{I}$ = set of all products with names starting with "FAUX" (from the data).
- For each $i \in \mathcal{I}$:
    - $A_i$ = revenue per unit of product $i$ (parameter from 'Revenue' column)
    - $d_i$ = demand for product $i$ (parameter from 'Demand' column)
    - $I_i$ = initial inventory for product $i$ (parameter from 'Initial Inventory' column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Subject to:**
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in \mathcal{I} \quad \text{(Demand constraint)} \\
& x_i \leq I_i, && \forall i \in \mathcal{I} \quad \text{(Inventory constraint)} \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, && \forall i \in \mathcal{I} \quad \text{(Nonnegativity and integrality)}
\end{align*}
\]

#### Data Mapping

- Table ID: file_0_view_0 (from ZARASales.csv)
    - Index set $\mathcal{I}$: All rows where 'Product Name' starts with "FAUX"
    - Parameter $A_i$: 'Revenue' column
    - Parameter $d_i$: 'Demand' column
    - Parameter $I_i$: 'Initial Inventory' column

No additional constraints or resource bounds are specified in the query. All parameters and index sets are mapped directly from the filtered data.