##### Objective Function:

$\quad \min \sum_{i=1}^6 \sum_{j=1}^6 c_{ij} x_{ij}$

where $c_{ij}$ is the cost of assigning manager $i$ to project $j$, and $x_{ij}$ is a binary variable equal to 1 if manager $i$ is assigned to project $j$, 0 otherwise.

##### Constraints

###### 1. Assignment Constraints:

Each manager is assigned to exactly one project:
$$
\sum_{j=1}^6 x_{ij} = 1 \quad \forall i \in \{1,2,3,4,5,6\}
$$

Each project is assigned to exactly one manager:
$$
\sum_{i=1}^6 x_{ij} = 1 \quad \forall j \in \{1,2,3,4,5,6\}
$$

###### 2. Variable Constraints:

$$
x_{ij} \in \{0,1\} \quad \forall i,j
$$

##### Retrieved Information

Managers (in source order):  
MA, MB, MC, MD, ME, MF

Projects (in source order):  
P1, P2, P3, P4, P5, P6

Cost matrix $c_{ij}$ (rows: managers, columns: projects):

|      | P1   | P2   | P3   | P4   | P5   | P6   |
|------|------|------|------|------|------|------|
| MA   | 2216 | 1911 | 1661 | 2122 | 1442 | 1442 |
| MB   | 1100 | 1271 | 2764 | 2557 | 1036 | 1036 |
| MC   | 2827 | 2784 | 2206 | 2216 | 2677 | 2677 |
| MD   | 2627 | 1273 | 2610 | 1957 | 1594 | 1594 |
| ME   | 3359 | 1003 | 2554 | 1706 | 2065 | 2065 |
| MF   | 1579 | 2289 | 2368 | 1922 | 2740 | 2740 |

Decision variables:
- $x_{ij} = 1$ if manager $i$ (MA, MB, MC, MD, ME, MF) is assigned to project $j$ (P1, P2, P3, P4, P5, P6), 0 otherwise.

The model assigns each manager to exactly one project and each project to exactly one manager, minimizing the total assignment cost.