##### Objective Function:

$\quad \min \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}$

where $c_{ij}$ is the cost of assigning Manager $i$ to Project $j$, and $x_{ij}$ is a binary variable indicating whether Manager $i$ is assigned to Project $j$.

##### Constraints:

###### 1. Each manager is assigned to exactly one project:

$\sum_{j=1}^{11} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,11\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i=1}^{11} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,11\}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Data Mapping

- Managers: 
  - Manager 1
  - Manager 2
  - Manager 3
  - Manager 4
  - Manager 5
  - Manager 6
  - Manager 7
  - Manager 8
  - Manager 9
  - Manager 10
  - Manager 11

- Projects:
  - Project 1
  - Project 2
  - Project 3
  - Project 4
  - Project 5
  - Project 6
  - Project 7
  - Project 8
  - Project 9
  - Project 10
  - Project 11

- Cost matrix $c_{ij}$: $c_{ij}$ is the value in row "Manager $i$" and column "Project $j$ Cost" from the provided CSV data. For example, $c_{1,1}$ is the cost for Manager 1 to Project 1, $c_{2,3}$ is the cost for Manager 2 to Project 3, etc. All values are taken directly from the CSV columns and rows as indexed above.