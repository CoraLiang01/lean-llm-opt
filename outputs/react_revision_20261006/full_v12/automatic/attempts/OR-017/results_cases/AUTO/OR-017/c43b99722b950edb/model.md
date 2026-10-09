#### Symbolic Mathematical Model

Let:

- $I$ = index set of all products with SKU prefix ‘ZZ’ (from table_id file_0_view_0, column SKU)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (file_0_view_0, column Revenue)
    - $d_i$ = demand for product $i$ (file_0_view_0, column Demand)
    - $s_i$ = initial inventory of product $i$ (file_0_view_0, column Initial Inventory)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i x_i
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

- Index set $I$: All rows in table_id file_0_view_0 where SKU has prefix ‘ZZ’ (column SKU)
- $A_i$: file_0_view_0, column Revenue
- $d_i$: file_0_view_0, column Demand
- $s_i$: file_0_view_0, column Initial Inventory
- $x_i$: decision variable for each $i \in I$