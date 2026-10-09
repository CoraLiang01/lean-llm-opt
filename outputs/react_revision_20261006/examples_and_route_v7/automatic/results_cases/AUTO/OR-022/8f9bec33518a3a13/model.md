##### Mathematical Model

Let:
- $I$ = set of all products in file_0_view_0 whose "Product Name" contains "27in"
- For each $i \in I$:
    - $r_i$ = Revenue for product $i$ (from "Revenue" column)
    - $d_i$ = Demand for product $i$ (from "Demand" column)
    - $s_i$ = Initial Inventory for product $i$ (from "Initial Inventory" column)
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

- Index set $I$: All records in table_id file_0_view_0 where "Product Name" contains "27in"
- $r_i$: file_0_view_0, column "Revenue", for each $i \in I$
- $d_i$: file_0_view_0, column "Demand", for each $i \in I$
- $s_i$: file_0_view_0, column "Initial Inventory", for each $i \in I$
- Decision variable $x_i$: number of units of product $i$ to fulfill, for each $i \in I$