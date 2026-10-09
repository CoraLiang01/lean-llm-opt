##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (binary)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$ (if supplier $i$ is not open, it cannot supply any store)
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from fixed_cost.csv, file_1_view_0, column "Unnamed: 0")
- $J$ = set of stores (from transportation_costs.csv, file_2_view_0, columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT")
- $d_j$ = demand for store $j$ (from demand.csv, file_0_view_0, column "demand", mapped to $J$)
- $f_i$ = fixed cost for supplier $i$ (from fixed_cost.csv, file_1_view_0, column "fixed_costs")
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, file_2_view_0, row "Unnamed: 0" = $i$, column $j$)
- $D_j$ = demand for store $j$ (from demand.csv, file_0_view_0, column "demand"); used as a valid upper bound for $x_{ij}$ in constraint 2

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 0"
- $J$: file_2_view_0, columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"
- $d_j$: file_0_view_0, column "demand", mapped to $J$
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $D_j$: file_0_view_0, column "demand", mapped to $J$