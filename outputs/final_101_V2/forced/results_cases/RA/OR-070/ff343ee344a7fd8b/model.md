##### Sets and Indices

Let $M$ be the set of managers:
- Manager 1
- Manager 2
- Manager 3
- Manager 4
- Manager 5
- Manager 6
- Manager 7

Let $P$ be the set of projects:
- Project 1
- Project 2
- Project 3
- Project 4
- Project 5
- Project 6
- Project 7

Let $c_{mp}$ be the cost for manager $m$ to complete project $p$, as given below.

##### Parameters

The cost matrix $c_{mp}$ is:

|             | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 | Project 6 | Project 7 |
|-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Manager 1   |   2972    |   2727    |   2795    |   2922    |   1302    |   2489    |   1533    |
| Manager 2   |   1094    |   2158    |   2990    |   1844    |   2887    |   2021    |   2288    |
| Manager 3   |   2133    |   1675    |   2422    |   2639    |   1033    |   2261    |   1695    |
| Manager 4   |   1951    |   2309    |   2070    |   2802    |   2328    |   1313    |   2434    |
| Manager 5   |   1269    |   2153    |   1296    |   2685    |   2627    |   1610    |   1641    |
| Manager 6   |   1220    |   1192    |   2907    |   2622    |   2595    |   1261    |   2384    |
| Manager 7   |   1286    |   1659    |   1179    |   1348    |   1420    |   2862    |   1959    |

##### Decision Variables

Let $x_{mp} = \begin{cases} 1 & \text{if manager } m \text{ is assigned to project } p \\ 0 & \text{otherwise} \end{cases}$

##### Objective Function

Minimize the total assignment cost:
$$
\min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}
$$

##### Constraints

1. Each project is assigned to exactly one manager:
$$
\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
$$

2. Each manager is assigned to at most one project:
$$
\sum_{p \in P} x_{mp} \leq 1 \quad \forall m \in M
$$

3. Binary assignment variables:
$$
x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P
$$

##### Complete Model

Minimize
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
- $x_{1,1} + x_{2,1} + x_{3,1} + x_{4,1} + x_{5,1} + x_{6,1} + x_{7,1} = 1$
- $x_{1,2} + x_{2,2} + x_{3,2} + x_{4,2} + x_{5,2} + x_{6,2} + x_{7,2} = 1$
- $x_{1,3} + x_{2,3} + x_{3,3} + x_{4,3} + x_{5,3} + x_{6,3} + x_{7,3} = 1$
- $x_{1,4} + x_{2,4} + x_{3,4} + x_{4,4} + x_{5,4} + x_{6,4} + x_{7,4} = 1$
- $x_{1,5} + x_{2,5} + x_{3,5} + x_{4,5} + x_{5,5} + x_{6,5} + x_{7,5} = 1$
- $x_{1,6} + x_{2,6} + x_{3,6} + x_{4,6} + x_{5,6} + x_{6,6} + x_{7,6} = 1$
- $x_{1,7} + x_{2,7} + x_{3,7} + x_{4,7} + x_{5,7} + x_{6,7} + x_{7,7} = 1$

- $x_{1,1} + x_{1,2} + x_{1,3} + x_{1,4} + x_{1,5} + x_{1,6} + x_{1,7} \leq 1$
- $x_{2,1} + x_{2,2} + x_{2,3} + x_{2,4} + x_{2,5} + x_{2,6} + x_{2,7} \leq 1$
- $x_{3,1} + x_{3,2} + x_{3,3} + x_{3,4} + x_{3,5} + x_{3,6} + x_{3,7} \leq 1$
- $x_{4,1} + x_{4,2} + x_{4,3} + x_{4,4} + x_{4,5} + x_{4,6} + x_{4,7} \leq 1$
- $x_{5,1} + x_{5,2} + x_{5,3} + x_{5,4} + x_{5,5} + x_{5,6} + x_{5,7} \leq 1$
- $x_{6,1} + x_{6,2} + x_{6,3} + x_{6,4} + x_{6,5} + x_{6,6} + x_{6,7} \leq 1$
- $x_{7,1} + x_{7,2} + x_{7,3} + x_{7,4} + x_{7,5} + x_{7,6} + x_{7,7} \leq 1$

- $x_{mp} \in \{0,1\}$ for all managers $m$ and projects $p$.