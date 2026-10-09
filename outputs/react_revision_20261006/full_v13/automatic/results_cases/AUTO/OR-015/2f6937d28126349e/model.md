#### Symbolic Mathematical Model

Let:

- $I$ = index set of all products classified under ‘Aalop’ (from the data, all products where "Product Name" has prefix "Aalop")
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = total demand for product $i$ over the sales horizon (parameter from column "Demand")
    - $I_i$ = initial inventory of product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory constraints:
$$
x_i \leq I_i \quad \forall i \in I
$$

- Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

- Variable domain:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- Table: RestaurantSalesreport.csv
    - Index set $I$: All rows where "Product Name" has prefix "Aalop" (table_id: file_0_view_0, column: "Product Name")
    - Parameter $A_i$: "Revenue" column (table_id: file_0_view_0, column: "Revenue")
    - Parameter $d_i$: "Demand" column (table_id: file_0_view_0, column: "Demand")
    - Parameter $I_i$: "Initial Inventory" column (table_id: file_0_view_0, column: "Initial Inventory")