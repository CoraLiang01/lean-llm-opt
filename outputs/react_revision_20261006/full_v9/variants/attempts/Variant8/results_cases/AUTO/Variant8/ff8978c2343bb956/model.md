##### Decision Variables

- $y_i \in \{0,1\}$: 1 if clinic $i$ is opened, 0 otherwise, for all clinics $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to clinic $i$, 0 otherwise, for all clinics $i \in I$ and neighborhoods $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}
\]

where:
- $d_j$ = demand of neighborhood $j$
- $c_{ij}$ = travel distance from clinic $i$ to neighborhood $j$

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one clinic:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

2. **Clinic Opening Limit:** Exactly $p$ clinics are opened:
   \[
   \sum_{i \in I} y_i = p
   \]
   where $p$ is the required number of clinics to open.

3. **Assignment Only to Opened Clinics:** Neighborhoods can only be assigned to open clinics:
   \[
   x_{ij} \leq y_i, \quad \forall i \in I, \forall j \in J
   \]

4. **Variable Domains:**
   \[
   x_{ij} \in \{0,1\}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- Clinics $I$: All unique values in column `"Clinic"` of `clinic_distance.csv` (`file_1_view_0`).
- Neighborhoods $J$: All unique values in column `"Neighborhood"` of `neighborhood_demand.csv` (`file_0_view_0`).
- Demand $d_j$: `"Demand"` column in `neighborhood_demand.csv` (`file_0_view_0`), indexed by `"Neighborhood"`.
- Distance $c_{ij}$: Entry in `clinic_distance.csv` (`file_1_view_0`), row `"Clinic"`, column for neighborhood $j$.
- Number of clinics to open $p$: Value in `planning_parameters.csv` (`file_2_view_0`), row where `"Parameter"` = `"NumberOfClinicsToOpen"`, column `"Value"`.

##### Data Mapping

- $I$: `file_1_view_0`, column `"Clinic"`
- $J$: `file_0_view_0`, column `"Neighborhood"`
- $d_j$: `file_0_view_0`, columns `"Neighborhood"`, `"Demand"`
- $c_{ij}$: `file_1_view_0`, row `"Clinic"`, columns `"N1"`–`"N10"`
- $p$: `file_2_view_0`, row `"Parameter" = "NumberOfClinicsToOpen"`, column `"Value"`