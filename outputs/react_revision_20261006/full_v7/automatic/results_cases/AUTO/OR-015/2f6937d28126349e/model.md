#### Symbolic Mathematical Model

Let:
- $I$ = set of all products classified under ‘Aalop’ (from the data, $I = \{\text{Aalopuri}\}$)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column ‘Revenue’)
    - $d_i$ = demand for product $i$ (parameter from column ‘Demand’)
    - $I_i$ = initial inventory of product $i$ (parameter from column ‘Initial Inventory’)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
x_i \leq d_i \quad \forall i \in I
\]
\[
x_i \leq I_i \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

#### Data Mapping

- Table: file_0_view_0 (RestaurantSalesreport.csv)
    - Index set $I$: All rows where ‘Product Name’ has prefix ‘Aalop’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $I_i$: column ‘Initial Inventory’
    - Variable $x_i$: number of units of product $i$ to fulfill (decision variable for each $i \in I$)