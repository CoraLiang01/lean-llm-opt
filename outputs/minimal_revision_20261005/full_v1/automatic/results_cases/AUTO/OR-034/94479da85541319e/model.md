#### Symbolic Optimization Model

Let:
- $I$ = index set of all baked goods (from column ‘Product Name’ in table_id: file_0_view_0)
- For each $i \in I$:
    - $A_i$ = revenue per unit of baked good $i$ (from column ‘Revenue’)
    - $d_i$ = total demand for baked good $i$ (from column ‘Demand’)
    - $s_i$ = initial inventory of baked good $i$ (from column ‘Initial Inventory’)
    - $x_i$ = quantity of baked good $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Demand and inventory limits:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I
$$

- Variable domain:
$$
x_i \in \mathbb{Z} \qquad \forall i \in I
$$

#### Data Mapping

- Index set $I$: All rows in table_id: file_0_view_0, column ‘Product Name’
- Parameter $A_i$: table_id: file_0_view_0, column ‘Revenue’
- Parameter $d_i$: table_id: file_0_view_0, column ‘Demand’
- Parameter $s_i$: table_id: file_0_view_0, column ‘Initial Inventory’
- Variable $x_i$: Decision variable for each $i \in I$