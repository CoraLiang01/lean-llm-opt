#### Symbolic Optimization Model

Let:
- $I$ = index set of all products classified under ‘Baby’ (from column ‘Product Name’ in table_id file_0_view_0)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column ‘Revenue’ in table_id file_0_view_0)
    - $d_i$ = demand for product $i$ (from column ‘Demand’ in table_id file_0_view_0)
    - $s_i$ = initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id file_0_view_0)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory constraints: $x_i \leq s_i \quad \forall i \in I$
- Demand constraints:  $x_i \leq d_i \quad \forall i \in I$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

#### Data Mapping

- Index set $I$ and product identifiers: table_id file_0_view_0, column ‘Product Name’
- Revenue parameter $A_i$: table_id file_0_view_0, column ‘Revenue’
- Demand parameter $d_i$: table_id file_0_view_0, column ‘Demand’
- Initial inventory parameter $s_i$: table_id file_0_view_0, column ‘Initial Inventory’