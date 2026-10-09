##### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from "Product Name" in file_0_view_0)
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ ("Revenue", file_0_view_0)
    - $d_i$ = demand for product $i$ ("Demand", file_0_view_0)
    - $s_i$ = initial inventory of product $i$ ("Initial Inventory", file_0_view_0)
- Decision variable: $x_i$ = number of units of product $i$ to fulfill

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I
$$
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

##### Data Mapping

- $I$: All products from file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Initial Inventory", keyed by "Product Name"
- $x_i$: Decision variable for each $i \in I$ (product "Product Name")

All parameters and index sets are defined directly from the returned data.