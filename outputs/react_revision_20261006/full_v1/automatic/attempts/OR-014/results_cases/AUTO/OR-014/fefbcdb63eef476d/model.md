#### Symbolic Mathematical Model

Let:

- $I$ = set of all pizza types (from the dataset)
- For each $i \in I$:
    - $A_i$ = revenue per unit of pizza type $i$ (parameter, from column "Revenue")
    - $d_i$ = total demand for pizza type $i$ (parameter, from column "Demand")
    - $s_i$ = initial inventory for pizza type $i$ (parameter, from column "Initial Inventory")
    - $x_i$ = number of units of pizza type $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- $I$: All unique values in column "Product Name" from table_id: file_0_view_0
- $A_i$: "Revenue" column, table_id: file_0_view_0
- $d_i$: "Demand" column, table_id: file_0_view_0
- $s_i$: "Initial Inventory" column, table_id: file_0_view_0

All parameters are mapped directly from the specified columns in table_id: file_0_view_0 (PizzaSalesDataset.csv).