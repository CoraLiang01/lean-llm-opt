##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand:  
$\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Plant capacity:  
$\sum_{j \in J} x_{ij} \leq \text{cap}_i y_i, \quad \forall i \in I$

3. Variable domains:  
$x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of plants, $I = \{\text{facility\_id}\}$ from column "facility_id" in table_id file_0_view_0 (cost.csv)
- $J$: set of customers, $J = \{\text{customer\_id}\}$ from column "customer_id" in table_id file_1_view_0 (demand.csv)
- $f_i$: fixed opening cost for plant $i$, from column "fixed_opening_cost" in table_id file_0_view_0
- $\text{cap}_i$: capacity of plant $i$, from column "facility_capacity" in table_id file_0_view_0
- $d_j$: demand of customer $j$, from column "demand_units" in table_id file_1_view_0
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from column "transportation_cost_to_{j}" in table_id file_0_view_0, where $j$ matches customer_id in file_1_view_0

##### Data Mapping

- Plants: file_0_view_0, column "facility_id"
- Plant fixed costs: file_0_view_0, column "fixed_opening_cost"
- Plant capacities: file_0_view_0, column "facility_capacity"
- Customers: file_1_view_0, column "customer_id"
- Customer demands: file_1_view_0, column "demand_units"
- Transportation costs: file_0_view_0, columns "transportation_cost_to_C1", ..., "transportation_cost_to_C15" (one per customer, matching customer_id in file_1_view_0)

All indices, parameters, and constraints are defined directly from the current CSV data.