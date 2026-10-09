##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is activated (open).

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$ (inactive suppliers cannot ship)
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from file_1_view_0: column "Unnamed: 2")
- $J$ = set of stores (from file_0_view_0: column "Customer")
- $d_j$ = demand of store $j$ (from file_0_view_0: column "demand")
- $f_i$ = fixed cost for supplier $i$ (from file_1_view_0: column "fixed_costs")
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from file_2_view_0: row "Unnamed: 2" = $i$, column = $j$'s city name)
- $D_j$ = demand of store $j$ (from file_0_view_0: column "demand$), used as a valid upper bound for $x_{ij}$ if $y_i=0$

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 2"
- $J$: file_0_view_0, column "Customer"
- $d_j$: file_0_view_0, column "demand"
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 2" = $i$, column = city name of $j$
- $x_{ij}$, $y_i$: decision variables as defined above

All index sets, parameters, and coefficients are mapped directly from the current CSV files as described.