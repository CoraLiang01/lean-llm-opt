##### Mathematical Model

Let:
- $I$ = set of products with id_number prefix 'id999' (from column id_number in file_0_view_0)
- For each $i \in I$:
    - $r_i$ = Revenue per unit of product $i$ (from Revenue column)
    - $d_i$ = Demand for product $i$ during the sales horizon (from Demand column)
    - $s_i$ = Initial Inventory of product $i$ (from Initial Inventory column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable; non-negative integer)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
$$
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Data Mapping

- Index set $I$: All rows in OnlineRetailSalesDataset.csv (file_0_view_0) where id_number has prefix 'id999'
- $r_i$: Revenue column, file_0_view_0, for each $i$
- $d_i$: Demand column, file_0_view_0, for each $i$
- $s_i$: Initial Inventory column, file_0_view_0, for each $i$
- $x_i$: Decision variable for each $i \in I$ (non-negative integer)