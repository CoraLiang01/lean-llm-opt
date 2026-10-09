#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with "Product Name" starting with "Baby" in table_id file_0_view_0.
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Inventory constraint: $x_i \leq s_i \quad \forall i \in I$
- Demand constraint: $x_i \leq d_i \quad \forall i \in I$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where "Product Name" starts with "Baby"
- $A_i$: "Revenue" column in file_0_view_0
- $d_i$: "Demand" column in file_0_view_0
- $s_i$: "Initial Inventory" column in file_0_view_0
- All data from: EuropeSalesRecords.csv, table_id file_0_view_0, columns ["Product Name", "Revenue", "Demand", "Initial Inventory"]