##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Plant capacity:  
   $\sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I$

3. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of plants, from column facility_id in cost.csv (table_id: file_0_view_0)
- $J$: set of customers, from column customer_id in demand.csv (table_id: file_1_view_0)
- $f_i$: fixed opening cost for plant $i$, from column fixed_opening_cost in cost.csv (table_id: file_0_view_0)
- $u_i$: capacity of plant $i$, from column facility_capacity in cost.csv (table_id: file_0_view_0)
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from columns transportation_cost_to_C1, ..., transportation_cost_to_C15 in cost.csv (table_id: file_0_view_0), with $j$ matched to customer_id in demand.csv
- $d_j$: demand of customer $j$, from column demand_units in demand.csv (table_id: file_1_view_0)

##### Data Mapping

- Plants $I$: file_0_view_0, column facility_id
- Customers $J$: file_1_view_0, column customer_id
- Fixed opening cost $f_i$: file_0_view_0, column fixed_opening_cost
- Plant capacity $u_i$: file_0_view_0, column facility_capacity
- Transportation cost $c_{ij}$: file_0_view_0, columns transportation_cost_to_C1 ... transportation_cost_to_C15, with $j$ matched to customer_id in file_1_view_0
- Demand $d_j$: file_1_view_0, column demand_units

All indices, parameters, and constraints are defined symbolically and mapped to their exact source columns and table_ids.