#### Abstract Mathematical Model

Let:

- $I$ = index set of all products classified as ‘27in’ (from the data, indexed by $i \in I$)
- $A_i$ = revenue per unit of product $i$ (parameter, from column ‘Revenue’)
- $d_i$ = deterministic demand for product $i$ (parameter, from column ‘Demand’)
- $s_i$ = initial inventory for product $i$ (parameter, from column ‘Initial Inventory’)
- $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \quad \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{+}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: Salesorders.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where ‘Product Name’ has prefix ‘27in’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’