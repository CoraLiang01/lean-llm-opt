#### Symbolic Mathematical Model

Let:
- $I$ = set of all products with ‘Organ’ in the Sub Category (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from Revenue column)
    - $d_i$ = demand for product $i$ (parameter from Demand column)
    - $s_i$ = initial inventory for product $i$ (parameter from Initial Inventory column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} A_i x_i
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

- Table: file_0_view_0 (SupermartGrocerySales-RetailAnalyticsDataset.csv)
    - Index set $I$: All rows where Sub Category starts with "Organ"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Decision variable $x_i$: number of units fulfilled for each $i \in I$