Let $x_{mp}$ be a binary variable equal to 1 if manager $m$ is assigned to project $p$, and 0 otherwise.

**Sets and Indices:**
- $M$: set of managers = {Manager 1, Manager 2, Manager 3, Manager 4, Manager 5, Manager 6, Manager 7}
- $P$: set of projects = {Project 1, Project 2, Project 3, Project 4, Project 5, Project 6, Project 7}

**Parameters:**
Let $c_{mp}$ be the cost for manager $m$ to complete project $p$, as given below:

|              | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 | Project 6 | Project 7 |
|--------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Manager 1    |   2972    |   2727    |   2795    |   2922    |   1302    |   2489    |   1533    |
| Manager 2    |   1094    |   2158    |   2990    |   1844    |   2887    |   2021    |   2288    |
| Manager 3    |   2133    |   1675    |   2422    |   2639    |   1033    |   2261    |   1695    |
| Manager 4    |   1951    |   2309    |   2070    |   2802    |   2328    |   1313    |   2434    |
| Manager 5    |   1269    |   2153    |   1296    |   2685    |   2627    |   1610    |   1641    |
| Manager 6    |   1220    |   1192    |   2907    |   2622    |   2595    |   1261    |   2384    |
| Manager 7    |   1286    |   1659    |   1179    |   1348    |   1420    |   2862    |   1959    |

**Objective:**
\[
\min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}
\]

**Subject to:**

1. **Each project is assigned to exactly one manager:**
   \[
   \sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
   \]

2. **Each manager is assigned to exactly one project:**
   \[
   \sum_{p \in P} x_{mp} = 1 \quad \forall m \in M
   \]

3. **Binary assignment variables:**
   \[
   x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P
   \]

**Where:**
- $x_{mp}$ = 1 if manager $m$ is assigned to project $p$, 0 otherwise.
- $c_{mp}$ = cost for manager $m$ to complete project $p$, as given in the table above.