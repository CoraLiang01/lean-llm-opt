##### Objective Function:

$\quad \min \sum_{i=1}^7 \sum_{j=1}^7 c_{ij} x_{ij}$

where $c_{ij}$ is the cost for Manager $i$ to be assigned to Project $j$, and $x_{ij}$ is a binary variable indicating whether Manager $i$ is assigned to Project $j$.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j=1}^7 x_{ij} = 1 \quad \forall i \in \{1,2,3,4,5,6,7\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i=1}^7 x_{ij} = 1 \quad \forall j \in \{1,2,3,4,5,6,7\}$

###### 3. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Data Mapping

- Managers: ["Manager 1", "Manager 2", "Manager 3", "Manager 4", "Manager 5", "Manager 6", "Manager 7"]
- Projects: ["Project 1", "Project 2", "Project 3", "Project 4", "Project 5", "Project 6", "Project 7"]
- Cost matrix $c_{ij}$: $c_{ij}$ is the value in row $i$ (Manager $i$) and column $j$ (Project $j$) of the CSV, as follows:

| $i$ (Manager) | $j$ (Project) | $c_{ij}$ (Cost) | CSV Column |
|---------------|---------------|-----------------|------------|
| 1 (Manager 1) | 1             | 2972            | Project 1 Cost |
| 1             | 2             | 2727            | Project 2 Cost |
| 1             | 3             | 2795            | Project 3 Cost |
| 1             | 4             | 2922            | Project 4 Cost |
| 1             | 5             | 1302            | Project 5 Cost |
| 1             | 6             | 2489            | Project 6 Cost |
| 1             | 7             | 1533            | Project 7 Cost |
| 2 (Manager 2) | 1             | 1094            | Project 1 Cost |
| 2             | 2             | 2158            | Project 2 Cost |
| 2             | 3             | 2990            | Project 3 Cost |
| 2             | 4             | 1844            | Project 4 Cost |
| 2             | 5             | 2887            | Project 5 Cost |
| 2             | 6             | 2021            | Project 6 Cost |
| 2             | 7             | 2288            | Project 7 Cost |
| 3 (Manager 3) | 1             | 2133            | Project 1 Cost |
| 3             | 2             | 1675            | Project 2 Cost |
| 3             | 3             | 2422            | Project 3 Cost |
| 3             | 4             | 2639            | Project 4 Cost |
| 3             | 5             | 1033            | Project 5 Cost |
| 3             | 6             | 2261            | Project 6 Cost |
| 3             | 7             | 1695            | Project 7 Cost |
| 4 (Manager 4) | 1             | 1951            | Project 1 Cost |
| 4             | 2             | 2309            | Project 2 Cost |
| 4             | 3             | 2070            | Project 3 Cost |
| 4             | 4             | 2802            | Project 4 Cost |
| 4             | 5             | 2328            | Project 5 Cost |
| 4             | 6             | 1313            | Project 6 Cost |
| 4             | 7             | 2434            | Project 7 Cost |
| 5 (Manager 5) | 1             | 1269            | Project 1 Cost |
| 5             | 2             | 2153            | Project 2 Cost |
| 5             | 3             | 1296            | Project 3 Cost |
| 5             | 4             | 2685            | Project 4 Cost |
| 5             | 5             | 2627            | Project 5 Cost |
| 5             | 6             | 1610            | Project 6 Cost |
| 5             | 7             | 1641            | Project 7 Cost |
| 6 (Manager 6) | 1             | 1220            | Project 1 Cost |
| 6             | 2             | 1192            | Project 2 Cost |
| 6             | 3             | 2907            | Project 3 Cost |
| 6             | 4             | 2622            | Project 4 Cost |
| 6             | 5             | 2595            | Project 5 Cost |
| 6             | 6             | 1261            | Project 6 Cost |
| 6             | 7             | 2384            | Project 7 Cost |
| 7 (Manager 7) | 1             | 1286            | Project 1 Cost |
| 7             | 2             | 1659            | Project 2 Cost |
| 7             | 3             | 1179            | Project 3 Cost |
| 7             | 4             | 1348            | Project 4 Cost |
| 7             | 5             | 1420            | Project 5 Cost |
| 7             | 6             | 2862            | Project 6 Cost |
| 7             | 7             | 1959            | Project 7 Cost |

- $x_{ij}$: Binary decision variable, $x_{ij} = 1$ if Manager $i$ is assigned to Project $j$, $0$ otherwise.