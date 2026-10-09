##### Objective Function:

$\quad \min \sum_{i=1}^7 \sum_{j=1}^7 c_{ij} x_{ij}$

where $c_{ij}$ is the cost for Manager $i$ to complete Project $j$, and $x_{ij}$ is a binary variable equal to 1 if Manager $i$ is assigned to Project $j$, 0 otherwise.

##### Constraints

###### 1. Assignment Constraints:

Each manager is assigned to exactly one project:
$$
\sum_{j=1}^7 x_{ij} = 1 \quad \forall i \in \{1,2,3,4,5,6,7\}
$$

Each project is assigned to exactly one manager:
$$
\sum_{i=1}^7 x_{ij} = 1 \quad \forall j \in \{1,2,3,4,5,6,7\}
$$

###### 2. Variable Constraints:

$$
x_{ij} \in \{0,1\} \quad \forall i,j
$$

##### Retrieved Information

Managers:  
- Manager 1  
- Manager 2  
- Manager 3  
- Manager 4  
- Manager 5  
- Manager 6  
- Manager 7  

Projects:  
- Project 1  
- Project 2  
- Project 3  
- Project 4  
- Project 5  
- Project 6  
- Project 7  

Cost matrix $c_{ij}$ (rows: managers, columns: projects):

|               | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 | Project 6 | Project 7 |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Manager 1     |   2972    |   2727    |   2795    |   2922    |   1302    |   2489    |   1533    |
| Manager 2     |   1094    |   2158    |   2990    |   1844    |   2887    |   2021    |   2288    |
| Manager 3     |   2133    |   1675    |   2422    |   2639    |   1033    |   2261    |   1695    |
| Manager 4     |   1951    |   2309    |   2070    |   2802    |   2328    |   1313    |   2434    |
| Manager 5     |   1269    |   2153    |   1296    |   2685    |   2627    |   1610    |   1641    |
| Manager 6     |   1220    |   1192    |   2907    |   2622    |   2595    |   1261    |   2384    |
| Manager 7     |   1286    |   1659    |   1179    |   1348    |   1420    |   2862    |   1959    |

##### Decision Variables

$x_{ij} = \begin{cases}
1 & \text{if Manager } i \text{ is assigned to Project } j \\
0 & \text{otherwise}
\end{cases}$

##### Model Summary

- Minimize total assignment cost.
- Each manager is assigned to exactly one project.
- Each project is assigned to exactly one manager.
- All assignments are binary decisions.

All cost and identifier data are preserved as retrieved.