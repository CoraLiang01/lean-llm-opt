#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products classified under ‘Fashion’ (from Product Name in file_0_view_0)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from Revenue)
    - $d_i$ = deterministic demand for product $i$ (from Demand)
    - $s_i$ = initial inventory for product $i$ (from Initial Inventory)
    - $x_i$ = integer number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Inventory and demand fulfillment limits:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I
$$
- Integer variables:
$$
x_i \in \mathbb{Z} \qquad \forall i \in I
$$

#### Data Mapping

- $I$: All rows in table_id file_0_view_0 where Product Name starts with "Fashion"
- $A_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $s_i$: file_0_view_0, column "Initial Inventory"
- $x_i$: decision variable for each $i \in I$