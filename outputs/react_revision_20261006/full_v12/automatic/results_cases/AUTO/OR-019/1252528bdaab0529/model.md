#### Symbolic Mathematical Model

Let:

- $I$ = set of all products with "Product Name" starting with "27in" (from table_id file_0_view_0)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column "Revenue")
    - $d_i$ = demand for product $i$ (from column "Demand")
    - $I_i$ = initial inventory for product $i$ (from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
$$
x_i \leq d_i \quad \forall i \in I
$$

$$
x_i \leq I_i \quad \forall i \in I
$$

$$
x_i \geq 0 \quad \forall i \in I
$$

$$
x_i \in \mathbb{Z} \quad \forall i \in I
$$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where "Product Name" starts with "27in"
- Parameter $A_i$: "Revenue" column in table_id file_0_view_0
- Parameter $d_i$: "Demand" column in table_id file_0_view_0
- Parameter $I_i$: "Initial Inventory" column in table_id file_0_view_0
- Variable $x_i$: Decision variable for each $i \in I$ (units fulfilled for product $i$)