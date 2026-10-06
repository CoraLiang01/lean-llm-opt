#### Symbolic Mathematical Model

Let:

- $I$ = index set of all “4U” products (from the data)
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
- Inventory and demand constraints for each $i \in I$:
    $$
    0 \leq x_i \leq \min\{d_i,\, s_i\}
    $$
    (or, equivalently, two constraints:)
    $$
    x_i \leq d_i \qquad \forall i \in I
    $$
    $$
    x_i \leq s_i \qquad \forall i \in I
    $$
    $$
    x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
    $$

#### Data Mapping

- Table ID: file_0_view_0 (from OnlineSalesinUSA.csv)
    - Index set $I$: All rows where "Product Name" starts with "4U"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"