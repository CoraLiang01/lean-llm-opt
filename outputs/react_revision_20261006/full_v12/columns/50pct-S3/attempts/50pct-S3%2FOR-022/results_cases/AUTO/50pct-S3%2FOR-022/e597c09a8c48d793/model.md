##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Branch demand:  
$\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Supplier activation:  
$\sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I$  
where $M_i = \sum_{j \in J} d_j$ (since there are no explicit supplier capacity limits)

3. Domains:  
$x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of suppliers, from column "facility_id" in table_id "file_1_view_0" and "file_2_view_0"
- $J$: set of branches, from column "customer_id" in table_id "file_0_view_0"
- $d_j$: demand of branch $j$, from column "demand_units" in table_id "file_0_view_0"
- $f_i$: fixed opening cost for supplier $i$, from column "fixed_opening_cost" in table_id "file_1_view_0"
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from column "transportation_cost_to_{j}" in table_id "file_2_view_0" (with $j$ mapped as above)
- $M_i$: big-M for each supplier $i$, set to $\sum_{j \in J} d_j$ (total demand), as no explicit supplier capacity limits are given

##### Data Mapping

- $I$: "facility_id" in "file_1_view_0" and "file_2_view_0"
- $J$: "customer_id" in "file_0_view_0"
- $d_j$: "demand_units" in "file_0_view_0"
- $f_i$: "fixed_opening_cost" in "file_1_view_0"
- $c_{ij}$: "transportation_cost_to_{j}" in "file_2_view_0", with $j$ as in $J$
- $M_i$: $\sum_{j \in J} d_j$ (computed from "demand_units" in "file_0_view_0")