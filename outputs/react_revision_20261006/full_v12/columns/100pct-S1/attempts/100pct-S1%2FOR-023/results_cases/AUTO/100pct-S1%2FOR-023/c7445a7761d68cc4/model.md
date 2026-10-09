##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$ (if a supplier is not open, it cannot ship to any store)
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from file_1_view_0, column "Unnamed: 2")
- $J$ = set of stores (from file_0_view_0, column "Customer")
- $d_j$ = demand of store $j$ (from file_0_view_0, column "demand")
- $f_i$ = fixed cost for supplier $i$ (from file_1_view_0, column "fixed_costs")
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from file_2_view_0, row "Unnamed: 2" = $i$, column header = $j$)
- $D_j$ = demand of store $j$ (from file_0_view_0, column "demand$); this is a valid upper bound for $x_{ij}$

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 2"
- $J$: file_0_view_0, column "Customer"
- $d_j$: file_0_view_0, column "demand"
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 2" = $i$, column = $j$ (column names must be mapped from store names in $J$ to columns in file_2_view_0)
- $D_j$: file_0_view_0, column "demand"

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files.