#### Mathematical Optimization Model

Let:
- $I$ = set of all products classified under ‘Aalop’ (from the data, $I = \{\text{Aalopuri}\}$)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from ‘Revenue’ column)
    - $d_i$ = demand for product $i$ (parameter from ‘Demand’ column)
    - $s_i$ = initial inventory of product $i$ (parameter from ‘Initial Inventory’ column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
x_i \leq d_i \quad \forall i \in I \qquad \text{(Demand constraint)}
\]
\[
x_i \leq s_i \quad \forall i \in I \qquad \text{(Inventory constraint)}
\]
\[
x_i \geq 0 \quad \forall i \in I \qquad \text{(Non-negativity, integer if required)}
\]

#### Data Mapping

- Table: RestaurantSalesreport.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where ‘Product Name’ has prefix ‘Aalop’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’