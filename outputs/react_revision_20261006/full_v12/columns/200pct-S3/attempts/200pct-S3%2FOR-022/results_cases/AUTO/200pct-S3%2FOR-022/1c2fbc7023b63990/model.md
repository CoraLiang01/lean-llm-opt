##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Branch demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Supplier activation logic:  
   $x_{ij} \leq U_{ij} y_i, \quad \forall i \in I, \forall j \in J$  
   (where $U_{ij}$ is a sufficiently large upper bound, e.g., $U_{ij} = d_j$)

3. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of suppliers (facility_id) from file_1_view_0 and file_2_view_0
- $J$: set of branches (customer_id) from file_0_view_0 and file_2_view_0
- $d_j$: demand for branch $j$ from column demand_units in file_0_view_0
- $f_i$: fixed opening cost for supplier $i$ from column fixed_opening_cost in file_1_view_0
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$ from columns transportation_cost_to_C* in file_2_view_0
- $U_{ij}$: upper bound for $x_{ij}$, set to $d_j$ (from file_0_view_0) for each $j$

##### Data Mapping

- $I$: facility_id in file_1_view_0 and file_2_view_0
- $J$: customer_id in file_0_view_0 and suffix of transportation_cost_to_C* in file_2_view_0
- $d_j$: file_0_view_0, column demand_units, indexed by customer_id
- $f_i$: file_1_view_0, column fixed_opening_cost, indexed by facility_id
- $c_{ij}$: file_2_view_0, columns transportation_cost_to_C*, indexed by facility_id and customer_id
- $U_{ij}$: $d_j$ from file_0_view_0, for each $j$

All index sets and parameters are defined by the full set of entities in the current CSV files.