##### Objective Function:

$\quad \min \sum_{i=1}^{12} \sum_{j=1}^{12} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^{12} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,12\}$

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,12\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Data Mapping

- Machines: $i \in \{$M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12$\}$
- Tasks: $j \in \{$A, B, C, D, E, F, G, H, I, J, K, L$\}$
- Cost coefficients: $c_{ij}$ is the cost of assigning machine $i$ to task $j$, as given in the columns of the CSV file, with $c_{ij}$ corresponding to the value in row "Machine" $=$ $i$, column $j$.

- Decision variables: $x_{ij} = 1$ if machine $i$ is assigned to task $j$, $0$ otherwise.