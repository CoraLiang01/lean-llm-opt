#### Symbolic Mathematical Model

Let:

- $I$ = index set of all products with names starting with "S700_" (from SampleSalesData.csv)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $s_i$ = initial inventory of product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i \, x_i
$$

Subject to:
- Demand and inventory limits:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
$$
- Integrality:
$$
x_i \in \mathbb{Z} \quad \forall i \in I
$$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where "Product Name" has prefix "S700_"
- $A_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $s_i$: file_0_view_0, column "Initial Inventory"
- $x_i$: decision variable for each $i \in I$