#### Mathematical Optimization Model

Let:
- $I$ = set of all products with "27in" in their Product Name from the dataset.
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = demand for product $i$ (parameter)
    - $s_i$ = initial inventory for product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i x_i
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

- Index set $I$: All rows in table_id file_0_view_0 (SalesDataAnalysis.csv) where column "Product Name" contains "27in".
- Parameter $A_i$: file_0_view_0, column "Revenue", for each $i \in I$.
- Parameter $d_i$: file_0_view_0, column "Demand", for each $i \in I$.
- Parameter $s_i$: file_0_view_0, column "Initial Inventory", for each $i \in I$.
- Decision variable $x_i$: defined for each $i \in I$.

All data is sourced from table_id file_0_view_0, columns "Product Name", "Revenue", "Demand", "Initial Inventory".