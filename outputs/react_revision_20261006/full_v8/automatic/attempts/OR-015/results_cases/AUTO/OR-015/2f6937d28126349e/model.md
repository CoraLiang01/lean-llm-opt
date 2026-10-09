#### Symbolic Mathematical Model

Let:
- $I$ = set of all products classified under ‘Aalop’ (from the data, $I = \{\text{Aalopuri}\}$)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = demand for product $i$ over the sales horizon (parameter)
    - $s_i$ = initial inventory of product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
x_i \leq d_i \qquad \forall i \in I \tag{Demand constraint}
\]
\[
x_i \leq s_i \qquad \forall i \in I \tag{Inventory constraint}
\]
\[
x_i \geq 0 \qquad \forall i \in I \tag{Nonnegativity}
\]

#### Data Mapping

- Table: RestaurantSalesreport.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where Product Name has prefix "Aalop"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"