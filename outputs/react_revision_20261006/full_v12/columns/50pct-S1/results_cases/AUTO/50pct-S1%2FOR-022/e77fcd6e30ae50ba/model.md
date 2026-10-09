##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Branch demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $\sum_{j \in J} x_{ij} \leq M_i y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (facility_id from file_1_view_0)
- $J$ = set of branches/customers (customer_id from file_0_view_0)
- $d_j$ = demand of branch $j$ (demand_units from file_0_view_0)
- $f_i$ = fixed opening cost for supplier $i$ (fixed_opening_cost from file_1_view_0)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to branch $j$ (transportation_cost_to_C* columns from file_2_view_0)
- $M_i$ = $\sum_{j \in J} d_j$ (total demand; valid upper bound since no supplier capacity is specified)

##### Data Mapping

- $I$: file_1_view_0, column facility_id
- $J$: file_0_view_0, column customer_id
- $d_j$: file_0_view_0, column demand_units, indexed by customer_id
- $f_i$: file_1_view_0, column fixed_opening_cost, indexed by facility_id
- $c_{ij}$: file_2_view_0, row facility_id, columns transportation_cost_to_C* (where * matches customer_id)
- $M_i$: $\sum_{j \in J} d_j$ (sum over file_0_view_0, column demand_units)

All index sets and parameters are defined by the full current contents of the respective CSV files.