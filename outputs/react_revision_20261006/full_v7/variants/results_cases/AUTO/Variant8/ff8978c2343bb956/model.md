##### Decision Variables

- $y_i \in \{0,1\}$: 1 if clinic $i$ is opened, 0 otherwise, for all clinics $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to clinic $i$, 0 otherwise, for all clinics $i \in I$, neighborhoods $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}
\]

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one clinic:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]
2. **Clinic opening limit:** Exactly $p$ clinics are opened:
   \[
   \sum_{i \in I} y_i = p
   \]
   where $p$ is the required number of clinics to open.
3. **Assignment only to open clinics:** Neighborhoods can only be assigned to open clinics:
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]
4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\},\quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of clinics, from column "Clinic" in table_id file_1_view_0 (clinic_distance.csv)
- $J$: set of neighborhoods, from column "Neighborhood" in table_id file_0_view_0 (neighborhood_demand.csv)
- $d_j$: demand of neighborhood $j$, from column "Demand" in table_id file_0_view_0 (neighborhood_demand.csv)
- $c_{ij}$: travel distance from clinic $i$ to neighborhood $j$, from entry at row $i$ (Clinic) and column $j$ (Neighborhood) in table_id file_1_view_0 (clinic_distance.csv)
- $p$: number of clinics to open, from row where "Parameter" = "NumberOfClinicsToOpen" and column "Value" in table_id file_2_view_0 (planning_parameters.csv)

##### Data Mapping

- Clinics $I$: file_1_view_0, column "Clinic"
- Neighborhoods $J$: file_0_view_0, column "Neighborhood"
- Demand $d_j$: file_0_view_0, column "Demand"
- Distance $c_{ij}$: file_1_view_0, row "Clinic" $i$, column $j$ (neighborhood ID)
- Number of clinics to open $p$: file_2_view_0, row "Parameter" = "NumberOfClinicsToOpen", column "Value"