##### Decision Variables

Let $x_{ij} \in \{0,1\}$ be a binary variable, where $x_{ij} = 1$ if Manager $i$ is assigned to Project $j$, and $x_{ij} = 0$ otherwise.

##### Parameters

Let $c_{ij}$ be the cost for Manager $i$ to complete Project $j$, as given in the table below:

|            | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 | Project 6 | Project 7 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Manager 1  | 2972      | 2727      | 2795      | 2922      | 1302      | 2489      | 1533      |
| Manager 2  | 1094      | 2158      | 2990      | 1844      | 2887      | 2021      | 2288      |
| Manager 3  | 2133      | 1675      | 2422      | 2639      | 1033      | 2261      | 1695      |
| Manager 4  | 1951      | 2309      | 2070      | 2802      | 2328      | 1313      | 2434      |
| Manager 5  | 1269      | 2153      | 1296      | 2685      | 2627      | 1610      | 1641      |
| Manager 6  | 1220      | 1192      | 2907      | 2622      | 2595      | 1261      | 2384      |
| Manager 7  | 1286      | 1659      | 1179      | 1348      | 1420      | 2862      | 1959      |

##### Objective Function

\[
\min \sum_{i=1}^{7} \sum_{j=1}^{7} c_{ij} x_{ij}
\]

##### Constraints

1. Each project is assigned to exactly one manager:
   \[
   \sum_{i=1}^{7} x_{ij} = 1, \quad \forall j = 1, \ldots, 7
   \]

2. Each manager is assigned to exactly one project:
   \[
   \sum_{j=1}^{7} x_{ij} = 1, \quad \forall i = 1, \ldots, 7
   \]

3. Binary assignment variables:
   \[
   x_{ij} \in \{0,1\}, \quad \forall i = 1, \ldots, 7; \quad j = 1, \ldots, 7
   \]

##### Sets

- Managers: $\{ \text{Manager 1}, \text{Manager 2}, \text{Manager 3}, \text{Manager 4}, \text{Manager 5}, \text{Manager 6}, \text{Manager 7} \}$
- Projects: $\{ \text{Project 1}, \text{Project 2}, \text{Project 3}, \text{Project 4}, \text{Project 5}, \text{Project 6}, \text{Project 7} \}$

All cost coefficients $c_{ij}$ are as given in the table above. The model assigns each manager to exactly one project and each project to exactly one manager, minimizing the total assignment cost.