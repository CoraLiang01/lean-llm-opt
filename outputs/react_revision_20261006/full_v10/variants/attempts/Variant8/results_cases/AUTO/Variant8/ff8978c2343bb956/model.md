##### Decision Variables

$y_i \in \{0,1\}$: 1 if clinic $i \in I$ is opened, 0 otherwise.

$x_{ij} \in \{0,1\}$: 1 if neighborhood $j \in J$ is assigned to clinic $i \in I$, 0 otherwise.

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}$

##### Constraints

1. Assignment: $\sum_{i \in I} x_{ij} = 1,\quad \forall j \in J$
2. Open clinic limit: $\sum_{i \in I} y_i = p$
3. Assignment only to open clinics: $x_{ij} \leq y_i,\quad \forall i \in I,\, j \in J$
4. Binary variables: $x_{ij} \in \{0,1\},\ y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of candidate clinics (Clinic column in clinic_distance.csv, table_id: file_1_view_0)
- $J$: set of neighborhoods (Neighborhood column in neighborhood_demand.csv, table_id: file_0_view_0)
- $d_j$: demand of neighborhood $j$ (Demand column in neighborhood_demand.csv, table_id: file_0_view_0)
- $c_{ij}$: travel distance from clinic $i$ to neighborhood $j$ (column $j$ in clinic_distance.csv, table_id: file_1_view_0, row $i$)
- $p$: number of clinics to open (Value where Parameter = NumberOfClinicsToOpen in planning_parameters.csv, table_id: file_2_view_0)

##### Data Mapping

- $I$: file_1_view_0, column "Clinic"
- $J$: file_0_view_0, column "Neighborhood"
- $d_j$: file_0_view_0, column "Demand"
- $c_{ij}$: file_1_view_0, row "Clinic", columns "N1"..."N10"
- $p$: file_2_view_0, row where Parameter = "NumberOfClinicsToOpen", column "Value"