##### Decision Variables

$y_i \in \{0,1\}$: 1 if facility location $i$ is opened, 0 otherwise, for all $i \in I$.

$x_{ij} \in \{0,1\}$: 1 if area $j$ is assigned to facility location $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}$

where:
- $d_j$ = demand of area $j$
- $c_{ij}$ = distance from location $i$ to area $j$

##### Constraints

1. **Assignment:** Each area is assigned to exactly one facility:
   $$
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   $$

2. **Facility count:** Exactly $p$ facilities are opened:
   $$
   \sum_{i \in I} y_i = p
   $$
   where $p$ is the required number of facilities to open.

3. **Assignment-to-open-facility linking:** Areas can only be assigned to open facilities:
   $$
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   $$

4. **Binary restrictions:**
   $$
   x_{ij} \in \{0,1\},\quad y_i \in \{0,1\},\quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$: set of candidate facility locations, from column "Location" in table_id "file_1_view_0" (location_distance.csv)
- $J$: set of areas, from column "Area" in table_id "file_0_view_0" (area_demand.csv)
- $d_j$: demand for area $j$, from column "Demand" in table_id "file_0_view_0" (area_demand.csv)
- $c_{ij}$: distance from location $i$ to area $j$, from entry at row $i$ (column "Location") and column $j$ (area name) in table_id "file_1_view_0" (location_distance.csv)
- $p$: number of facilities to open, from row where "Parameter" = "NumberOfFacilitiesToOpen", column "Value" in table_id "file_2_view_0" (planning_parameters.csv)