##### Mathematical Model

Let:
- $I$ = set of all products with 'Product Name' starting with "FAUX" (from table_id file_0_view_0)
- For each $i \in I$:
    - $r_i$ = Revenue for product $i$ (column 'Revenue')
    - $d_i$ = Demand for product $i$ (column 'Demand')
    - $s_i$ = Initial Inventory for product $i$ (column 'Initial Inventory')
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
$$
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0, column 'Product Name'
- Parameter $r_i$: table_id file_0_view_0, column 'Revenue'
- Parameter $d_i$: table_id file_0_view_0, column 'Demand'
- Parameter $s_i$: table_id file_0_view_0, column 'Initial Inventory'
- Variable $x_i$: number of units of product $i$ to fulfill, for each $i \in I$