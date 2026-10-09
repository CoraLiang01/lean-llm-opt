#### Mathematical Optimization Model

Let:
- $I$ = index set of all products with Product_Reference starting with "ELE-S"
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter, from column "Demand")
    - $s_i$ = initial inventory of product $i$ (parameter, from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory constraints:
$$
x_i \leq s_i \quad \forall i \in I
$$

- Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

- Variable domain:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where Product_Reference has prefix "ELE-S" (column "Product_Reference", file SalesStoreoverview.csv)
- Parameter $A_i$: column "Revenue", table_id file_0_view_0, file SalesStoreoverview.csv
- Parameter $d_i$: column "Demand", table_id file_0_view_0, file SalesStoreoverview.csv
- Parameter $s_i$: column "Initial Inventory", table_id file_0_view_0, file SalesStoreoverview.csv
- Decision variable $x_i$: defined for each $i \in I$ as above

No additional constraints or bounds are imposed beyond those specified above.