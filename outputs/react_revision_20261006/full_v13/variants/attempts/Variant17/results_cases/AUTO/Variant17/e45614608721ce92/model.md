##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility location $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if area $j$ is assigned to facility $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Parameters

- $d_j$: demand of area $j \in J$ (from area_demand.csv, column "Demand", table_id: file_0_view_0).
- $c_{ij}$: distance from facility $i$ to area $j$ (from location_distance.csv, table_id: file_1_view_0, row "Location" $i$, column $j$).
- $p$: number of facilities to open (from planning_parameters.csv, Parameter = "NumberOfFacilitiesToOpen", Value column, table_id: file_2_view_0).
- $I$: set of candidate facility locations (from location_distance.csv, "Location" column, table_id: file_1_view_0).
- $J$: set of areas (from area_demand.csv, "Area" column, table_id: file_0_view_0).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each area is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = p
   \]

3. **Assignment-to-Open-Facility Linking:** Areas can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]

4. **Variable Domains:**
   \[
   x_{ij} \in \{0,1\}, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

---

##### Data Mapping

- $I$: All values in "Location" column of location_distance.csv (table_id: file_1_view_0)
- $J$: All values in "Area" column of area_demand.csv (table_id: file_0_view_0)
- $d_j$: "Demand" column of area_demand.csv, indexed by "Area" (table_id: file_0_view_0)
- $c_{ij}$: location_distance.csv, row "Location" $i$, column $j$ (table_id: file_1_view_0)
- $p$: planning_parameters.csv, row where "Parameter" = "NumberOfFacilitiesToOpen", column "Value" (table_id: file_2_view_0)