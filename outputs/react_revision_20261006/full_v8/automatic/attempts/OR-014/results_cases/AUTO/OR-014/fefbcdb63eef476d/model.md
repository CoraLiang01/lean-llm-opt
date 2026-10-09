#### Symbolic Mathematical Model

Let:
- $I$ = index set of all pizza types (from Product Name)
- For each $i \in I$:
    - $A_i$ = revenue per unit of pizza type $i$ (from Revenue)
    - $d_i$ = total demand for pizza type $i$ (from Demand)
    - $s_i$ = initial inventory for pizza type $i$ (from Initial Inventory)
    - $x_i$ = number of units of pizza type $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- $I$: All rows in table_id = file_0_view_0, column = Product Name
- $A_i$: table_id = file_0_view_0, column = Revenue
- $d_i$: table_id = file_0_view_0, column = Demand
- $s_i$: table_id = file_0_view_0, column = Initial Inventory

All parameters are mapped directly from the specified columns in PizzaSalesDataset.csv.