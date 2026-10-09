##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Plant capacity:  
   $\sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I$

3. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of plants, from column "facility_id" in table_id file_0_view_0 (cost.csv)
- $J$: set of customers, from column "customer_id" in table_id file_1_view_0 (demand.csv)
- $f_i$: fixed opening cost for plant $i$, from column "fixed_opening_cost" in table_id file_0_view_0
- $u_i$: capacity of plant $i$, from column "facility_capacity" in table_id file_0_view_0
- $d_j$: demand of customer $j$, from column "demand_units" in table_id file_1_view_0
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from column "transportation_cost_to_{j}" in table_id file_0_view_0, where ${j}$ is the customer_id (e.g., "transportation_cost_to_C1" for $j = C1$)

##### Notes

- Each plant can only supply if it is built ($y_i = 1$), and its total shipments cannot exceed its capacity $u_i$.
- All customer demands must be exactly satisfied.
- The objective is to minimize the sum of fixed opening costs and transportation costs.

##### Data Mapping

- Plants: file_0_view_0, column "facility_id"
- Customers: file_1_view_0, column "customer_id"
- Fixed opening cost $f_i$: file_0_view_0, column "fixed_opening_cost"
- Plant capacity $u_i$: file_0_view_0, column "facility_capacity"
- Demand $d_j$: file_1_view_0, column "demand_units"
- Transportation cost $c_{ij}$: file_0_view_0, column "transportation_cost_to_{j}" (where ${j}$ is the customer_id from file_1_view_0)

No additional constraints are imposed beyond those described above.