##### Mathematical Model

Let:
- $I$ = set of all “4U” products (indexed by $i$; from column "Product Name" in table_id file_0_view_0)
- $r_i$ = revenue per unit of product $i$ (from column "Revenue")
- $d_i$ = demand for product $i$ (from column "Demand")
- $s_i$ = initial inventory of product $i$ (from column "Initial Inventory")
- $x_i$ = number of units of product $i$ to fulfill (decision variable; non-negative integer)

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

- $I$: All records in table_id file_0_view_0, column "Product Name"
- $r_i$: table_id file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: table_id file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: table_id file_0_view_0, column "Initial Inventory", keyed by "Product Name"
- $x_i$: Decision variable for each $i \in I$ (non-negative integer)