#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products classified under ‘Baby’ (from the data mapping)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = deterministic demand for product $i$ (parameter)
    - $s_i$ = initial inventory of product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad \forall i \in I \\
& x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table ID: file_0_view_0 (from Salesdata.csv)
    - Index set $I$: all rows where [Product Name] has prefix "Baby"
    - $A_i$: [Revenue] column
    - $d_i$: [Demand] column
    - $s_i$: [Initial Inventory] column

All parameters are mapped directly from the specified columns for each ‘Baby’ product.