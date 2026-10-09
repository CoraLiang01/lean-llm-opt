##### Decision Variables

$y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.

$x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Demand satisfaction at each distribution center:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$

2. Factory capacity (only if constructed):
   $$
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   $$

3. Variable domains:
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

##### Index Sets and Parameters

- $I$: set of factories, from facility_costs.csv, column Facility, table_id file_0_view_0.
- $J$: set of distribution centers, from demand_requirements.csv, column Destination, table_id file_2_view_0.
- $f_i$: fixed cost of constructing factory $i$, from facility_costs.csv, column FixedCost, table_id file_0_view_0.
- $K_i$: capacity of factory $i$, from facility_costs.csv, column Capacity, table_id file_0_view_0.
- $d_j$: demand at distribution center $j$, from demand_requirements.csv, column Demand, table_id file_2_view_0.
- $c_{ij}$: shipping cost per unit from factory $i$ to distribution center $j$, from shipping_costs.csv, row Origin = $i$, column $j$, table_id file_1_view_0.

##### Data Mapping

- Factories $I$: facility_costs.csv, column Facility, table_id file_0_view_0
- Distribution centers $J$: demand_requirements.csv, column Destination, table_id file_2_view_0
- Fixed costs $f_i$: facility_costs.csv, column FixedCost, table_id file_0_view_0
- Factory capacities $K_i$: facility_costs.csv, column Capacity, table_id file_0_view_0
- Demands $d_j$: demand_requirements.csv, column Demand, table_id file_2_view_0
- Shipping costs $c_{ij}$: shipping_costs.csv, row Origin = $i$, column $j$, table_id file_1_view_0

All index sets, parameters, and constraints are defined directly from the current CSV data.