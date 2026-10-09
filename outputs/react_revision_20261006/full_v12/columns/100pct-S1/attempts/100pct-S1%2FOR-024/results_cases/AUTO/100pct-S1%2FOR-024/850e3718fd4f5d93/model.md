##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Demand satisfaction: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Activation logic: $\sum_{j\in J} x_{ij} \leq M_i y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ = set of warehouses (from file_1_view_0, column "Unnamed: 0")
- $J$ = set of musicians/bands (from file_0_view_0, column "customer")
- $d_j$ = demand of musician/band $j$ (from file_0_view_0, column "demand")
- $f_i$ = fixed cost for warehouse $i$ (from file_1_view_0, column "fixed_costs")
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to musician/band $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)
- $M_i = \sum_{j\in J} d_j$ (a valid upper bound for total shipments from warehouse $i$)

##### Data Mapping

- Warehouses $I$: file_1_view_0, column "Unnamed: 0"
- Musicians/Bands $J$: file_0_view_0, column "customer"
- Demand $d_j$: file_0_view_0, column "demand"
- Fixed cost $f_i$: file_1_view_0, column "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $M_i$: $\sum_{j\in J} d_j$ (computed from file_0_view_0, column "demand")