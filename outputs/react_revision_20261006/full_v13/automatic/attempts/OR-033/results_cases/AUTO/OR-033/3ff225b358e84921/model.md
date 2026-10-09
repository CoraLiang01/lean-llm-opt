#### Mathematical Model

Let:
- $I$ = index set of all products with "Product Name" starting with "Baby" (from the data mapping below)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from "Revenue")
    - $d_i$ = demand for product $i$ (parameter from "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter from "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
1. Inventory constraints:
$$
x_i \leq s_i \quad \forall i \in I
$$

2. Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

3. Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in I
$$

#### Data Mapping

- Table: file_0_view_0 (EuropeSalesRecords.csv)
    - Index set $I$: All rows where "Product Name" starts with "Baby"
    - $A_i$: "Revenue" column
    - $d_i$: "Demand" column
    - $s_i$: "Initial Inventory" column

No additional constraints or parameters are imposed by the query or data.