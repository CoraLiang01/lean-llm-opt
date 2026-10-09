##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Supplier activation logic:  
   $\sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I$  
   where $M = \sum_{j \in J} d_j$ (a valid upper bound on total shipments from any supplier).

3. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of suppliers (from fixed_cost.csv, file_1_view_0, column "Unnamed: 1" and transportation_costs.csv, file_2_view_0, row "Unnamed: 0")
- $J$: set of stores (from demand.csv, file_0_view_0, column "Customer" and transportation_costs.csv, file_2_view_0, columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT")
- $d_j$: demand of store $j$ (from demand.csv, file_0_view_0, column "demand")
- $f_i$: fixed cost for supplier $i$ (from fixed_cost.csv, file_1_view_0, column "fixed_costs")
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, file_2_view_0, row "Unnamed: 0" for $i$, column $j$)
- $M = \sum_{j \in J} d_j$ (sum of all store demands, from demand.csv, file_0_view_0, column "demand")

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 1" and file_2_view_0, row "Unnamed: 0"
- $J$: file_0_view_0, column "Customer" and file_2_view_0, columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"
- $d_j$: file_0_view_0, column "demand"
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier $i$), column $j$
- $M$: $\sum_{j \in J} d_j$ from file_0_view_0, column "demand"