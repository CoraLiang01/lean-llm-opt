##### Symbolic Mathematical Model

Let:
- $I$ = set of all products $i$ classified under ‘Aalop’ (from the data, all products with "Product Name" prefix "Aalop")
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $I_i$ = initial inventory of product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Inventory constraint: $x_i \leq I_i \quad \forall i \in I$
- Demand constraint:  $x_i \leq d_i \quad \forall i \in I$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

##### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where "Product Name" has prefix "Aalop"
- Parameter $A_i$: "Revenue" column in table_id file_0_view_0
- Parameter $d_i$: "Demand" column in table_id file_0_view_0
- Parameter $I_i$: "Initial Inventory" column in table_id file_0_view_0
- Variable $x_i$: Decision variable for each $i \in I$