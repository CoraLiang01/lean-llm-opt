##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Branch demand:  
$\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Supplier activation:  
$x_{ij} \leq D_j y_i, \quad \forall i \in I, \forall j \in J$  
where $D_j$ is the demand of branch $j$ (from demand.csv).

3. Domains:  
$x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of suppliers (facility_id from fixed_cost.csv and transportation_costs.csv)
- $J$: set of branches (customer_id from demand.csv and transportation_costs.csv)
- $d_j$: demand of branch $j$ (demand_units from demand.csv, table_id: file_0_view_0, column: demand_units)
- $f_i$: fixed opening cost for supplier $i$ (fixed_opening_cost from fixed_cost.csv, table_id: file_1_view_0, column: fixed_opening_cost)
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$ (transportation_cost_to_C* columns from transportation_costs.csv, table_id: file_2_view_0)

##### Data Mapping

- $I$: facility_id from file_1_view_0 and file_2_view_0
- $J$: customer_id from file_0_view_0 and columns transportation_cost_to_C* in file_2_view_0
- $d_j$: file_0_view_0, column demand_units, indexed by customer_id
- $f_i$: file_1_view_0, column fixed_opening_cost, indexed by facility_id
- $c_{ij}$: file_2_view_0, columns transportation_cost_to_C*, rows facility_id; $c_{ij}$ is the entry in row $i$ (facility_id), column $j$ (transportation_cost_to_C*)
- $x_{ij}$, $y_i$: decision variables as defined above

All index sets and parameters are defined by the full set of entities in the current CSV files. No capacity limits are imposed beyond the activation constraint.