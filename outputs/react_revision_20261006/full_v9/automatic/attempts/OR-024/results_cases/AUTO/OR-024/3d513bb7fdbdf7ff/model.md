#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with names starting with "S700_" (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = demand for product $i$ (parameter)
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
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (SampleSalesData.csv)
    - Index set $I$: All rows where "Product Name" has prefix "S700_"
    - $A_i$: "Revenue" column
    - $d_i$: "Demand" column
    - $s_i$: "Initial Inventory" column

All parameters are mapped directly from the specified columns for each $i \in I$. Decision variables $x_i$ are integer and non-negative, as per the query.