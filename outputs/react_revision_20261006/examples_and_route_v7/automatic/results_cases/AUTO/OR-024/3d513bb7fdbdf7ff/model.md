##### Mathematical Model

Let:
- $I$ = set of products with names starting with "S700_" (indexed by $i$; see Data Mapping for exact IDs).
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$.
    - $d_i$ = demand for product $i$.
    - $s_i$ = initial inventory of product $i$.
    - $x_i$ = number of units of product $i$ to fulfill (decision variable).

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
- Demand fulfillment cannot exceed demand or available inventory:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
$$

- $x_i$ are nonnegative integers:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0, column "Product Name", with prefix "S700_".
- Parameter $r_i$: table_id file_0_view_0, column "Revenue", key "Product Name".
- Parameter $d_i$: table_id file_0_view_0, column "Demand", key "Product Name".
- Parameter $s_i$: table_id file_0_view_0, column "Initial Inventory", key "Product Name".
- Decision variable $x_i$: number of units to fulfill for product $i \in I$.

All data is mapped directly from the filtered rows of SampleSalesData.csv as described above.