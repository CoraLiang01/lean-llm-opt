#### Mathematical Optimization Model

Let:
- $I$ = index set of all products classified under ‘TABLET’ (from Product Name column)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from Revenue column)
    - $d_i$ = demand for product $i$ (from Demand column)
    - $s_i$ = initial inventory of product $i$ (from Initial Inventory column)
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

- Table: file_0_view_0 (SmartphoneRetailOutletSalesData.csv)
    - Index set $I$: All rows where Product Name starts with "TABLET"
    - $A_i$: Revenue column
    - $d_i$: Demand column
    - $s_i$: Initial Inventory column