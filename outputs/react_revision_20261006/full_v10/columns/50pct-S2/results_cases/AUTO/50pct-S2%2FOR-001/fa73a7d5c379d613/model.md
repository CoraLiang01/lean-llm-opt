##### Objective Function:

$\quad \min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in J} x_{ij} = 1 \quad \forall i \in I$

$\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in I, \forall j \in J$

##### Data Mapping

- $I$: Set of managers, from file_0_view_0 column "Unnamed: 0"
- $J$: Set of projects, from file_0_view_0 columns "P1", "P2", "P3"
- $c_{ij}$: Cost for manager $i$ to manage project $j$, from file_0_view_0, row "Unnamed: 0" = $i$, column $j$
- $x_{ij}$: Binary variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise

All data is mapped directly from table_id file_0_view_0, using the exact column and row identifiers as above.