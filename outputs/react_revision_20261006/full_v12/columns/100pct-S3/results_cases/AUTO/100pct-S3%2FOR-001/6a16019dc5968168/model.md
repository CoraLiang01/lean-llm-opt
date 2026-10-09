##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA},\text{MB},\text{MC}\}} \sum_{j \in \{\text{P1},\text{P2},\text{P3}\}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \{\text{P1},\text{P2},\text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA},\text{MB},\text{MC}\}$

$\sum_{i \in \{\text{MA},\text{MB},\text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1},\text{P2},\text{P3}\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA},\text{MB},\text{MC}\},\ j \in \{\text{P1},\text{P2},\text{P3}\}$

##### Data Mapping

- $i$ (manager): values from column "Unnamed: 0" in table_id "file_0_view_0"
- $j$ (project): columns "P1", "P2", "P3" in table_id "file_0_view_0"
- $c_{ij}$: entry in row with manager $i$ and column $j$ in table_id "file_0_view_0"
- $x_{ij}$: binary variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise

All sets and parameters are defined exactly as in the current CSV data.