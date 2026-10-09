#### Symbolic Mathematical Model

Let:
- $I$ = set of all pizza types (indexed by $i$)
- For each $i \in I$:
    - $A_i$ = revenue per unit of pizza type $i$ (parameter)
    - $d_i$ = total demand for pizza type $i$ (parameter)
    - $s_i$ = initial inventory for pizza type $i$ (parameter)
    - $x_i$ = number of units of pizza type $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (PizzaSalesDataset.csv)
    - Index set $I$: all unique values in column "Product Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"