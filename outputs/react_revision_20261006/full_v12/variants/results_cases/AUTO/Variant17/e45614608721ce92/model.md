##### Decision Variables

$y_i \in \{0,1\}$: 1 if facility at location $i$ is opened, 0 otherwise, for all $i \in I$.

$x_{ij} \in \{0,1\}$: 1 if area $j$ is assigned to facility at location $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Parameters

$I$: set of candidate facility locations (from column "Location" in table_id file_1_view_0).
$J$: set of areas (from column "Area" in table_id file_0_view_0).
$d_j$: demand of area $j$ (from column "Demand" in table_id file_0_view_0).
$c_{ij}$: distance from location $i$ to area $j$ (from entry [row $i$, column $j$] in table_id file_1_view_0).
$p$: number of facilities to open (from "Value" where "Parameter" = "NumberOfFacilitiesToOpen" in table_id file_2_view_0).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}$

##### Constraints

1. Each area assigned to exactly one facility:
   $$
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   $$

2. Exactly $p$ facilities are opened:
   $$
   \sum_{i \in I} y_i = p
   $$

3. Assignment only to open facilities:
   $$
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   $$

4. Binary restrictions:
   $$
   x_{ij} \in \{0,1\}, \quad \forall i \in I,\, j \in J
   $$
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$

---

###### Data Mapping

- $I$: All "Location" values in table_id file_1_view_0 (location_distance.csv)
- $J$: All "Area" values in table_id file_0_view_0 (area_demand.csv)
- $d_j$: "Demand" column for each "Area" in table_id file_0_view_0 (area_demand.csv)
- $c_{ij}$: Entry at [row "Location" = $i$, column $j$] in table_id file_1_view_0 (location_distance.csv)
- $p$: "Value" where "Parameter" = "NumberOfFacilitiesToOpen" in table_id file_2_view_0 (planning_parameters.csv)