##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Branch demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq U_{ij} y_i,\quad \forall i \in I,\, j \in J$ (where $U_{ij}$ is a valid upper bound, e.g., $U_{ij} = d_j$)
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of suppliers, from column facility_id in table_id file_1_view_0 and file_2_view_0
- $J$: set of branches, from column customer_id in table_id file_0_view_0 and mapped as columns in file_2_view_0
- $d_j$: demand of branch $j$, from column demand_units in table_id file_0_view_0
- $f_i$: fixed opening cost for supplier $i$, from column fixed_opening_cost in table_id file_1_view_0
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from columns transportation_cost_to_C* in table_id file_2_view_0, mapped to $j$
- $U_{ij}$: upper bound for $x_{ij}$, set to $d_j$ (branch demand) for each $i,j$

##### Data Mapping

- $I$: facility_id in file_1_view_0 and file_2_view_0
- $J$: customer_id in file_0_view_0 and suffix of transportation_cost_to_C* in file_2_view_0
- $d_j$: demand_units in file_0_view_0, indexed by customer_id
- $f_i$: fixed_opening_cost in file_1_view_0, indexed by facility_id
- $c_{ij}$: transportation_cost_to_C* in file_2_view_0, indexed by facility_id and mapped customer_id
- $U_{ij}$: $d_j$ from file_0_view_0

All index sets and parameters are defined directly from the current CSV data.