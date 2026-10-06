##### Decision Variables

- $y_i \in \{0,1\}$: 1 if clinic $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j \in J$ is assigned to clinic $i \in I$, 0 otherwise.

##### Parameters

- $d_j$: demand of neighborhood $j \in J$ (from **file_0_view_0**, column "Demand").
- $c_{ij}$: travel distance from clinic $i \in I$ to neighborhood $j \in J$ (from **file_1_view_0**, row "Clinic", columns "N1"–"N10").
- $p$: number of clinics to open (from **file_2_view_0**, column "Value" where "Parameter" = "NumberOfClinicsToOpen").

- $I$: set of candidate clinics (from **file_1_view_0**, column "Clinic").
- $J$: set of neighborhoods (from **file_0_view_0**, column "Neighborhood").

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:**  
   Each neighborhood is assigned to exactly one clinic:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Clinic Opening Limit:**  
   Exactly $p$ clinics are opened:
   \[
   \sum_{i \in I} y_i = p
   \]

3. **Assignment Only to Opened Clinics:**  
   Neighborhoods can only be assigned to open clinics:
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

#### Data Mapping

- $I$: all values in **file_1_view_0**, column "Clinic"
- $J$: all values in **file_0_view_0**, column "Neighborhood"
- $d_j$: **file_0_view_0**, columns "Neighborhood", "Demand"
- $c_{ij}$: **file_1_view_0**, row "Clinic", columns "N1"–"N10"
- $p$: **file_2_view_0**, column "Value" where "Parameter" = "NumberOfClinicsToOpen"