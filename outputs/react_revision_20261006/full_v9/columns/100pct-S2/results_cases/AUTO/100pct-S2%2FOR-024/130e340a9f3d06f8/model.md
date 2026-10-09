##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Musician/band demand: $\sum_{i\in I}x_{ij}=d_j,\quad \forall j\in J$.
2. Warehouse activation: $\sum_{j\in J}x_{ij}\leq M y_i,\quad \forall i\in I$.
3. Domains: $x_{ij}\geq0$ continuous; $y_i\in\{0,1\}$.

Where:
- $I$ = set of warehouses (from file_1_view_0: Unnamed: 0)
- $J$ = set of musicians/bands (from file_0_view_0: customer)
- $d_j$ = demand of musician/band $j$ (from file_0_view_0: demand)
- $f_i$ = fixed cost of warehouse $i$ (from file_1_view_0: fixed_costs)
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to musician/band $j$ (from file_2_view_0: row Unnamed: 0 = $i$, column = $j$)
- $M = \sum_{j\in J} d_j$ (total demand; unconditional upper bound for each warehouse)

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 0"
- $J$: file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand", indexed by "customer"
- $f_i$: file_1_view_0, column "fixed_costs", indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column = $j$
- $M$: $\sum_{j\in J} d_j$ from file_0_view_0, column "demand"