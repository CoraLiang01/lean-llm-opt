##### Mathematical Model

Let:
- $I$ = set of products with ‘Organ’ prefix in ‘Sub Category’ (indexed by $i$)
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ (from Revenue)
    - $d_i$ = demand for product $i$ (from Demand)
    - $s_i$ = initial inventory of product $i$ (from Initial Inventory)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
- Inventory constraint for each product:
$$
x_i \leq s_i \quad \forall i \in I
$$

- Demand constraint for each product:
$$
x_i \leq d_i \quad \forall i \in I
$$

- Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0, column ‘Sub Category’ with prefix ‘Organ’
- Parameter $r_i$: table_id file_0_view_0, column ‘Revenue’
- Parameter $d_i$: table_id file_0_view_0, column ‘Demand’
- Parameter $s_i$: table_id file_0_view_0, column ‘Initial Inventory’
- Variable $x_i$: number of units of product $i$ to fulfill, for each $i \in I$