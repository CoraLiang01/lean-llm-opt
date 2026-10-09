##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i$

##### Constraints

1. Supermarket demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq d_j y_i,\quad \forall i \in I,\, j \in J$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from file_1_view_0, column "Unnamed: 0" and file_2_view_0, column "Unnamed: 1")
- $J$ = set of supermarkets (from file_0_view_0, column "customer" and file_2_view_0, columns "C1", "C2")
- $d_j$ = demand of supermarket $j$ (from file_0_view_0, column "demand")
- $f_i$ = fixed cost for supplier $i$ (from file_1_view_0, column "fixed_costs")
- $c_{ij}$ = per-unit transportation cost from supplier $i$ to supermarket $j$ (from file_2_view_0, columns "C1", "C2" by row "Unnamed: 1")

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 0"; file_2_view_0, column "Unnamed: 1"
- $J$: file_0_view_0, column "customer"; file_2_view_0, columns "C1", "C2"
- $d_j$: file_0_view_0, column "demand"
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, columns "C1", "C2" (rows indexed by "Unnamed: 1")