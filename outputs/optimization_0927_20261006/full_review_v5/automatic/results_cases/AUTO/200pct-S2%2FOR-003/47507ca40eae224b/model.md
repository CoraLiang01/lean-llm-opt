##### Objective Function:

$\quad \min \sum_{i=1}^7 \sum_{j=1}^7 c_{ij} x_{ij}$

where $c_{ij}$ is the cost for assigning Manager $i$ to Project $j$, and $x_{ij}$ is a binary variable equal to 1 if Manager $i$ is assigned to Project $j$, 0 otherwise.

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

##### Complete Mathematical Model

Minimize:
$$
2972x_{1,1} + 2727x_{1,2} + 2795x_{1,3} + 2922x_{1,4} + 1302x_{1,5} + 2489x_{1,6} + 1533x_{1,7} \\
+ 1094x_{2,1} + 2158x_{2,2} + 2990x_{2,3} + 1844x_{2,4} + 2887x_{2,5} + 2021x_{2,6} + 2288x_{2,7} \\
+ 2133x_{3,1} + 1675x_{3,2} + 2422x_{3,3} + 2639x_{3,4} + 1033x_{3,5} + 2261x_{3,6} + 1695x_{3,7} \\
+ 1951x_{4,1} + 2309x_{4,2} + 2070x_{4,3} + 2802x_{4,4} + 2328x_{4,5} + 1313x_{4,6} + 2434x_{4,7} \\
+ 1269x_{5,1} + 2153x_{5,2} + 1296x_{5,3} + 2685x_{5,4} + 2627x_{5,5} + 1610x_{5,6} + 1641x_{5,7} \\
+ 1220x_{6,1} + 1192x_{6,2} + 2907x_{6,3} + 2622x_{6,4} + 2595x_{6,5} + 1261x_{6,6} + 2384x_{6,7} \\
+ 1286x_{7,1} + 1659x_{7,2} + 1179x_{7,3} + 1348x_{7,4} + 1420x_{7,5} + 2862x_{7,6} + 1959x_{7,7}
$$

Subject to:
- For each manager $i$ ($i=1,\ldots,7$): $\sum_{j=1}^7 x_{i,j} = 1$
- For each project $j$ ($j=1,\ldots,7$): $\sum_{i=1}^7 x_{i,j} = 1$
- $x_{i,j} \in \{0,1\}$ for all $i,j$