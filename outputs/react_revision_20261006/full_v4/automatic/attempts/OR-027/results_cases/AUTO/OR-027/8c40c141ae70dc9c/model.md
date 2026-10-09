#### Symbolic Mathematical Model

Let:
- $I$ = set of all products with ‘Organ’ in the Sub Category (from data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from Revenue column)
    - $d_i$ = demand for product $i$ (parameter from Demand column)
    - $s_i$ = initial inventory for product $i$ (parameter from Initial Inventory column)
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

- $I$: All rows in table_id = file_0_view_0, column = Sub Category, where Sub Category contains ‘Organ’
- $A_i$: table_id = file_0_view_0, column = Revenue
- $d_i$: table_id = file_0_view_0, column = Demand
- $s_i$: table_id = file_0_view_0, column = Initial Inventory
- $x_i$: Decision variable for each $i \in I$