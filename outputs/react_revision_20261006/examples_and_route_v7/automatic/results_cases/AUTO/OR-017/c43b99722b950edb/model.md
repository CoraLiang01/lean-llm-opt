Mathematical Model

Let $I$ be the set of all products in the dataset with SKU starting with 'ZZ' (i.e., all products classified under 'ZZ'). For each $i \in I$:

- $r_i$ = Revenue for product $i$ (parameter from column 'Revenue', table_id file_0_view_0, key SKU)
- $d_i$ = Demand for product $i$ (parameter from column 'Demand', table_id file_0_view_0, key SKU)
- $s_i$ = Initial Inventory for product $i$ (parameter from column 'Initial Inventory', table_id file_0_view_0, key SKU)
- $x_i$ = Number of units of product $i$ to fulfill (decision variable, nonnegative integer)

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

Data Mapping

- $I$: All SKUs in file_0_view_0 where SKU starts with 'ZZ'
- $r_i$: file_0_view_0, column 'Revenue', key SKU
- $d_i$: file_0_view_0, column 'Demand', key SKU
- $s_i$: file_0_view_0, column 'Initial Inventory', key SKU
- $x_i$: Decision variable for each $i \in I$ (SKU)

Constraints:

- For each $i \in I$, $x_i$ cannot exceed either the demand or the initial inventory for that product.
- $x_i$ must be a nonnegative integer.

Objective:

- Maximize total revenue from fulfilled units of 'ZZ' products.

Variables:

- $x_i$ is the number of units of product $i$ to fulfill, for each $i \in I$.

Index sets, parameters, and all mappings are as above.