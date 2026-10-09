##### Objective Function:

$\quad \min \sum_{i=1}^6 \sum_{j=1}^6 c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^6 x_{ij} = 1 \quad \forall i \in \{1,2,3,4,5,6\}$

$\sum_{i=1}^6 x_{ij} = 1 \quad \forall j \in \{1,2,3,4,5,6\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\}, \quad \forall i,j$

##### Data Mapping

- Managers (rows, $i$): MA, MB, MC, MD, ME, MF
- Projects (columns, $j$): P1, P2, P3, P4, P5, P6
- Cost coefficients $c_{ij}$: as given in "manager_project_costs.csv", where $c_{ij}$ is the cost of assigning manager $i$ to project $j$ according to the CSV table.