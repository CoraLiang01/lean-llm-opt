##### Decision Variables

$x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to facility $i$, 0 otherwise, for all $i \in I$, $j \in J$.

$y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise, for all $i \in I$.

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
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   $$

4. **Variable Domains:**
   $$
   x_{ij} \in \{0,1\}, \quad \forall i \in I,\, j \in J
   $$
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$

##### Data Mapping

- $I$: set of candidate facilities, from column "Facility" in table_id file_1_view_0 (facility_distance.csv)
- $J$: set of neighborhoods, from column "Neighborhood" in table_id file_0_view_0 (neighborhood_demand.csv)
- $d_j$: "Demand" column in table_id file_0_view_0, indexed by "Neighborhood"
- $c_{ij}$: entry in table_id file_1_view_0, row "Facility" $i$, column $j$ (neighborhood ID)
- $p$: "Value" where "Parameter" = "NumberOfFacilitiesToOpen" in table_id file_2_view_0 (planning_parameters.csv)