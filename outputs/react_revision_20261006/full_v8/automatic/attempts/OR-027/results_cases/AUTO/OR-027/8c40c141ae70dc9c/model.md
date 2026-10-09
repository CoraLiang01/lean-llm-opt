#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with ‘Organ’ in the Sub Category (from the data)
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
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where Sub Category has prefix "Organ"
- $A_i$: Revenue column in table_id file_0_view_0
- $d_i$: Demand column in table_id file_0_view_0
- $s_i$: Initial Inventory column in table_id file_0_view_0
- $x_i$: Decision variable for each $i \in I$ (Organ products)