#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products classified under ‘Aalop’ (from the data, all products with "Product Name" prefix "Aalop")
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $I_i$ = initial inventory of product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
1. Inventory constraint:
$$
x_i \leq I_i \quad \forall i \in I
$$

2. Demand constraint:
$$
x_i \leq d_i \quad \forall i \in I
$$

3. Variable domain:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- Table: file_0_view_0 (RestaurantSalesreport.csv)
    - Index set $I$: All rows where "Product Name" has prefix "Aalop"
    - $A_i$: "Revenue" column
    - $d_i$: "Demand" column
    - $I_i$: "Initial Inventory" column
    - $x_i$: Decision variable for each $i \in I$