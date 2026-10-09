##### Decision Variables

- $y_i \in \{0,1\}$: 1 if clinic $i$ is opened, 0 otherwise, for all clinics $i$ in the set $I$ (clinics from clinic_distance.csv).
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to clinic $i$, 0 otherwise, for all clinics $i \in I$ and neighborhoods $j \in J$ (neighborhoods from neighborhood_demand.csv).

##### Parameters

- $d_j$: demand of neighborhood $j \in J$ (from neighborhood_demand.csv).
- $c_{ij}$: travel distance from clinic $i$ to neighborhood $j$ (from clinic_distance.csv).
- $p$: number of clinics to open (from planning_parameters.csv, parameter "NumberOfClinicsToOpen").

##### Index Sets

- $I$: set of clinics (from the "Clinic" column in clinic_distance.csv).
- $J$: set of neighborhoods (from the "Neighborhood" column in neighborhood_demand.csv).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one clinic:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Open Clinic Limit:** Exactly $p$ clinics are opened:
   \[
   \sum_{i \in I} y_i = p
   \]

3. **Assignment Only to Opened Clinics:** A neighborhood can only be assigned to an open clinic:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

4. **Variable Domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

##### Data Mapping

- $I$: All values in column "Clinic" of table_id file_1_view_0 (clinic_distance.csv)
- $J$: All values in column "Neighborhood" of table_id file_0_view_0 (neighborhood_demand.csv)
- $d_j$: Column "Demand" in table_id file_0_view_0, indexed by "Neighborhood"
- $c_{ij}$: Entry in table_id file_1_view_0, row "Clinic" $i$, column $j$ (neighborhood)
- $p$: Value in table_id file_2_view_0, where "Parameter" = "NumberOfClinicsToOpen", column "Value"