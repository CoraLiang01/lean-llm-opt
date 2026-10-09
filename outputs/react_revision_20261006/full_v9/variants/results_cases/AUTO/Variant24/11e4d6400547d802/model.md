##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to facility $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}
\]

where:
- $I$ = set of candidate facilities (Facility, from facility_distance.csv)
- $J$ = set of neighborhoods (Neighborhood, from neighborhood_demand.csv)
- $d_j$ = demand of neighborhood $j$ (Demand, from neighborhood_demand.csv)
- $c_{ij}$ = distance from facility $i$ to neighborhood $j$ (facility_distance.csv, row Facility $i$, column $j$)

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = p
   \]
   where $p$ = NumberOfFacilitiesToOpen (Value, from planning_parameters.csv, Parameter = NumberOfFacilitiesToOpen)

3. **Assignment-to-Open-Facility Linking:** Neighborhoods can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]

4. **Binary Restrictions:**
   \[
   x_{ij} \in \{0,1\},\quad y_i \in \{0,1\},\quad \forall i \in I,\, j \in J
   \]

---

##### Data Mapping

- $I$: All Facility values from facility_distance.csv (table_id: file_1_view_0, column: Facility)
- $J$: All Neighborhood values from neighborhood_demand.csv (table_id: file_0_view_0, column: Neighborhood)
- $d_j$: Demand for neighborhood $j$ from neighborhood_demand.csv (table_id: file_0_view_0, columns: Neighborhood, Demand)
- $c_{ij}$: Distance from facility $i$ to neighborhood $j$ from facility_distance.csv (table_id: file_1_view_0, row: Facility $i$, column: $j$)
- $p$: Value from planning_parameters.csv where Parameter = NumberOfFacilitiesToOpen (table_id: file_2_view_0, columns: Parameter, Value)

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files.