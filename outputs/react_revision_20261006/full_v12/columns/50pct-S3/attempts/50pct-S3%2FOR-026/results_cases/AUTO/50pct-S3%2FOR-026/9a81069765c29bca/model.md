##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$
2. Plant capacity (only if built):
   $$
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   $$
3. Variable domains:
   $$
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   $$

##### Index Sets

- $I$: set of plants, from column facility_id in file_0_view_0 (cost.csv)
- $J$: set of customers, from column customer_id in file_1_view_0 (demand.csv)

##### Parameters and Data Mapping

- $f_i$: fixed opening cost for plant $i$ — file_0_view_0, column fixed_opening_cost, indexed by facility_id
- $K_i$: capacity of plant $i$ — file_0_view_0, column facility_capacity, indexed by facility_id
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$ — file_0_view_0, column transportation_cost_to_$j$, indexed by facility_id and customer_id (with $j$ mapped as C1...C15)
- $d_j$: demand of customer $j$ — file_1_view_0, column demand_units, indexed by customer_id

##### Data Mapping

- Plants $I$: file_0_view_0, column facility_id
- Customers $J$: file_1_view_0, column customer_id
- $f_i$: file_0_view_0, column fixed_opening_cost, indexed by facility_id
- $K_i$: file_0_view_0, column facility_capacity, indexed by facility_id
- $c_{ij}$: file_0_view_0, columns transportation_cost_to_C1 ... transportation_cost_to_C15, indexed by facility_id and customer_id
- $d_j$: file_1_view_0, column demand_units, indexed by customer_id

All indices, parameters, and constraints are defined directly from the provided CSV files.