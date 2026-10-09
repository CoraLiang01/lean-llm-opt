#### Symbolic Mathematical Model

Let:
- $I$ = set of all “4U” products (from the data, all products whose "Product Name" starts with "4U")
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
- Inventory and demand fulfillment constraints:
    $$
    0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
    $$
    (or equivalently, $x_i \leq d_i$, $x_i \leq s_i$, $x_i \in \mathbb{Z}_{\geq 0}$)

#### Data Mapping

- Table ID: file_0_view_0 (from OnlineSalesinUSA.csv)
    - Index set $I$: all rows where "Product Name" starts with "4U"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Variable $x_i$: defined for each $i \in I$