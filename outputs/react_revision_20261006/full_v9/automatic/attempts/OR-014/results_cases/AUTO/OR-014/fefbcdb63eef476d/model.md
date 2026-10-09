#### Symbolic Mathematical Model

Let:
- $I$ = index set of all pizza types (from the dataset)
- For each $i \in I$:
    - $A_i$ = revenue per unit of pizza type $i$ (parameter)
    - $d_i$ = total demand for pizza type $i$ (parameter)
    - $s_i$ = initial inventory for pizza type $i$ (parameter)
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

- Index set $I$: All unique values in column "Product Name" of table_id file_0_view_0 (PizzaSalesDataset.csv)
- Parameter $A_i$: "Revenue" column, table_id file_0_view_0, for each $i \in I$
- Parameter $d_i$: "Demand" column, table_id file_0_view_0, for each $i \in I$
- Parameter $s_i$: "Initial Inventory" column, table_id file_0_view_0, for each $i \in I$
- Decision variable $x_i$: defined for each $i \in I$ as above

No additional constraints or subsets are imposed by the query. All pizza types in the current dataset are included.