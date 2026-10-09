##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Plant capacity:  
   $\sum_{j \in J} x_{ij} \leq \text{cap}_i y_i, \quad \forall i \in I$

3. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets

$I$: set of plants (facility_id from file_0_view_0)  
$J$: set of customers (customer_id from file_1_view_0)

##### Parameter Mapping

- $f_i$: fixed_opening_cost for plant $i$ (facility_id) from column "fixed_opening_cost" in file_0_view_0
- $\text{cap}_i$: facility_capacity for plant $i$ (facility_id) from column "facility_capacity" in file_0_view_0
- $c_{ij}$: transportation_cost_to_Cj for plant $i$ (facility_id) and customer $j$ (customer_id) from columns "transportation_cost_to_C1", ..., "transportation_cost_to_C15" in file_0_view_0
- $d_j$: demand_units for customer $j$ (customer_id) from column "demand_units" in file_1_view_0

##### Data Mapping

- Plants $I$: facility_id in file_0_view_0
- Customers $J$: customer_id in file_1_view_0
- $f_i$: file_0_view_0, column "fixed_opening_cost", indexed by facility_id
- $\text{cap}_i$: file_0_view_0, column "facility_capacity", indexed by facility_id
- $c_{ij}$: file_0_view_0, columns "transportation_cost_to_C1" ... "transportation_cost_to_C15", indexed by facility_id and customer_id
- $d_j$: file_1_view_0, column "demand_units", indexed by customer_id

No parameters are omitted; all are mapped directly to the CSV source columns and IDs.