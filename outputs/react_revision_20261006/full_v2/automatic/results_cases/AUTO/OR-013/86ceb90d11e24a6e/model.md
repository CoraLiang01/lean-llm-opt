##### Symbolic Mathematical Model

Let:
- $I$ = set of all “4U” products (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $s_i$ = initial inventory of product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
$$
x_i \leq d_i \qquad \forall i \in I
$$

$$
x_i \leq s_i \qquad \forall i \in I
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0 where "Product Name" has prefix "4U"
- Parameter $A_i$: file_0_view_0, column "Revenue"
- Parameter $d_i$: file_0_view_0, column "Demand"
- Parameter $s_i$: file_0_view_0, column "Initial Inventory"