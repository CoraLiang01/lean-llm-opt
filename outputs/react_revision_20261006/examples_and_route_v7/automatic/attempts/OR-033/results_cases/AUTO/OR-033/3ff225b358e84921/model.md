##### Mathematical Model

Let:
- $I$ = set of all products with 'Product Name' starting with "Baby" (from Data Mapping below)
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = demand for product $i$ (parameter)
    - $s_i$ = initial inventory of product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
- Inventory constraint: $\quad x_i \leq s_i \quad \forall i \in I$
- Demand constraint: $\quad x_i \leq d_i \quad \forall i \in I$
- Nonnegativity and integrality: $\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0, column 'Product Name'
- $r_i$: file_0_view_0, column 'Revenue', key 'Product Name'
- $d_i$: file_0_view_0, column 'Demand', key 'Product Name'
- $s_i$: file_0_view_0, column 'Initial Inventory', key 'Product Name'
- $x_i$: Decision variable for each $i \in I$ (product with 'Product Name' starting with "Baby")