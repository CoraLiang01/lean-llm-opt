##### Mathematical Optimization Model

Let:
- $\mathcal{I}$ = index set of all car models classified under ‘FDK57’ (from Product Name column with prefix "FDK57" in the data).
- For each $i \in \mathcal{I}$:
    - $A_i$ = revenue per unit of car model $i$ (from Revenue column)
    - $d_i$ = deterministic demand for car model $i$ (from Demand column)
    - $I_i$ = initial inventory for car model $i$ (from Initial Inventory column)
    - $x_i$ = integer variable: quantity of car model $i$ to fulfill

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \quad \text{(Demand constraint)} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \quad \text{(Inventory constraint)} \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I} \quad \text{(Nonnegativity and integrality)}
\end{align*}
\]

##### Data Mapping

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $\mathcal{I}$: All rows where Product Name has prefix "FDK57"
    - $A_i$: Revenue column
    - $d_i$: Demand column
    - $I_i$: Initial Inventory column
    - $x_i$: Decision variable for each $i \in \mathcal{I}$

No additional constraints or synthetic scenario parameters are specified in the query. All bounds and parameters are mapped directly from the returned data.