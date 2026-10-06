##### Objective Function:

$\quad \min \sum_{i=1}^{12} \sum_{j=1}^{12} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^{12} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,12\}$

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,12\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Data Mapping

- Machines (rows): $i \in \{1,2,\ldots,12\}$, corresponding to Machine labels: M1, M2, ..., M12
- Tasks (columns): $j \in \{1,2,\ldots,12\}$, corresponding to Task labels: A, B, ..., L
- Cost matrix $c_{ij}$: $c_{ij}$ is the machining cost for assigning Machine $i$ (M1–M12) to Task $j$ (A–L), as specified in the columns and rows of cost_12x12.csv.

- Decision variable: $x_{ij} = 1$ if Machine $i$ is assigned to Task $j$, $0$ otherwise.