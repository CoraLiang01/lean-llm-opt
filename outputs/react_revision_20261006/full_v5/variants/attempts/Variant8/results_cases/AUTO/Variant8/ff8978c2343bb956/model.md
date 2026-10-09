##### Decision Variables

- $y_i \in \{0,1\}$: 1 if clinic $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j \in J$ is assigned to clinic $i \in I$, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}
\]

where:
- $d_j$ = demand of neighborhood $j$ (from Data Mapping below)
- $c_{ij}$ = travel distance from clinic $i$ to neighborhood $j$ (from Data Mapping below)

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one clinic:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Clinic opening limit:** Exactly $p$ clinics are opened:
   \[
   \sum_{i \in I} y_i = p
   \]
   where $p$ is the required number of clinics to open.

3. **Assignment only to open clinics:** A neighborhood can only be assigned to an open clinic:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Index Sets and Data Mapping

- $I$: Set of candidate clinics, from column "Clinic" in table_id file_1_view_0 (clinic_distance.csv)
- $J$: Set of neighborhoods, from column "Neighborhood" in table_id file_0_view_0 (neighborhood_demand.csv)
- $d_j$: Demand for neighborhood $j$, from column "Demand" in table_id file_0_view_0, indexed by "Neighborhood"
- $c_{ij}$: Distance from clinic $i$ to neighborhood $j$, from table_id file_1_view_0, row "Clinic" = $i$, column $j$
- $p$: Number of clinics to open, from table_id file_2_view_0, row where "Parameter" = "NumberOfClinicsToOpen", column "Value"

All parameters and sets are mapped directly to the source data as described above.