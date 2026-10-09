##### Decision Variables

$x_{ij} \in \{0,1\}$: 1 if neighborhood $j \in J$ is assigned to facility $i \in I$, 0 otherwise.

$y_i \in \{0,1\}$: 1 if facility $i \in I$ is opened, 0 otherwise.

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}$

where:
- $d_j$ = demand of neighborhood $j$
- $c_{ij}$ = distance from facility $i$ to neighborhood $j$

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one facility:
   $$
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   $$

2. **Facility Opening:** Exactly $p$ facilities are opened:
   $$
   \sum_{i \in I} y_i = p
   $$
   where $p$ is the required number of facilities to open.

3. **Assignment-to-Open-Facility Linking:** Neighborhoods can only be assigned to open facilities:
   $$
   x_{ij} \leq y_i, \quad \forall i \in I, \forall j \in J
   $$

4. **Variable Domains:**
   $$
   x_{ij} \in \{0,1\}, \quad \forall i \in I, \forall j \in J
   $$
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$

##### Index Sets and Parameter Mapping

- $I$: Set of candidate facilities, from column "Facility" in table_id file_1_view_0 (facility_distance.csv)
- $J$: Set of neighborhoods, from column "Neighborhood" in table_id file_0_view_0 (neighborhood_demand.csv)
- $d_j$: Demand for neighborhood $j$, from column "Demand" in table_id file_0_view_0 (neighborhood_demand.csv)
- $c_{ij}$: Distance from facility $i$ to neighborhood $j$, from table_id file_1_view_0 (facility_distance.csv), row "Facility" = $i$, column $j$
- $p$: Number of facilities to open, from row where "Parameter" = "NumberOfFacilitiesToOpen", column "Value" in table_id file_2_view_0 (planning_parameters.csv)