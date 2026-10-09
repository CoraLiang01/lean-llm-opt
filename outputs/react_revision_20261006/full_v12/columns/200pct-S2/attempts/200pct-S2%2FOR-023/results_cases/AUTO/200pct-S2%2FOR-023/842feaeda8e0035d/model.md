##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq U_{ij} y_i,\quad \forall i \in I,\, j \in J$ (where $U_{ij}$ is a sufficiently large upper bound, e.g., $U_{ij} = d_j$)
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of suppliers (facility names from column "Unnamed: 3" in file_1_view_0)
- $J$: set of stores (store names from columns "BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO" in file_2_view_0)
- $d_j$: demand for store $j$ (from column "demand" in file_0_view_0, mapped to $J$)
- $f_i$: fixed cost for supplier $i$ (from column "fixed_costs" in file_1_view_0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from file_2_view_0, row "Unnamed: 4" = $i$, column $j$)
- $U_{ij}$: upper bound for $x_{ij}$, set to $d_j$ for each $j$

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 3"
- $J$: file_2_view_0, columns ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"]
- $d_j$: file_0_view_0, column "demand", mapped to $J$
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 4" = $i$, column $j$
- $U_{ij}$: $d_j$ (from file_0_view_0, column "demand", mapped to $J$)

All index sets and parameters are defined directly from the current CSV data.