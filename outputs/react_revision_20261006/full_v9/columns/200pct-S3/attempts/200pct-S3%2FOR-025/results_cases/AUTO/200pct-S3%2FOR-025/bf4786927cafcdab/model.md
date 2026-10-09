##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Supermarket demand: $\sum_{i\in I}x_{ij}=d_j,\quad \forall j\in J$
2. Supplier activation: $\sum_{j\in J}x_{ij}\leq M_i y_i,\quad \forall i\in I$
3. Domains: $x_{ij}\geq0$ continuous; $y_i\in\{0,1\}$

Where:
- $I$ = set of suppliers (from file_1_view_0["Unnamed: 0"])
- $J$ = set of supermarkets (from file_0_view_0["customer"])
- $d_j$ = demand of supermarket $j$ (from file_0_view_0["demand"])
- $f_i$ = fixed cost for supplier $i$ (from file_1_view_0["fixed_costs"])
- $c_{ij}$ = per-unit transportation cost from supplier $i$ to supermarket $j$ (from file_2_view_0, rows indexed by supplier $i$ and columns by supermarket $j$)
- $M_i$ = $\sum_{j\in J} d_j$ (since no explicit supplier capacity is given, $M_i$ is the total demand and serves as a valid upper bound)

##### Data Mapping

- $I$: file_1_view_0["Unnamed: 0"]
- $J$: file_0_view_0["customer"]
- $d_j$: file_0_view_0["demand"], indexed by $j$
- $f_i$: file_1_view_0["fixed_costs"], indexed by $i$
- $c_{ij}$: file_2_view_0, rows indexed by "Unnamed: 1" (supplier $i$), columns by supermarket $j$ ("C1", "C2", etc.)
- $M_i$: $\sum_{j\in J} d_j$ (computed from file_0_view_0["demand"])