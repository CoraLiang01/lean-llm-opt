##### Decision Variables

$x_{ij} \in \{0,1\}$: 1 if manager $i$ is assigned to project $j$, 0 otherwise, for all managers $i \in M$ and projects $j \in P$.

##### Objective Function

$\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}$

##### Constraints

1. Each manager is assigned to exactly one project:
   $$
   \sum_{j \in P} x_{ij} = 1, \quad \forall i \in M
   $$
2. Each project is assigned to exactly one manager:
   $$
   \sum_{i \in M} x_{ij} = 1, \quad \forall j \in P
   $$
3. Binary assignment:
   $$
   x_{ij} \in \{0,1\}, \quad \forall i \in M,\, j \in P
   $$

##### Parameters

- Managers: $M = \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$
- Projects: $P = \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

- Assignment cost matrix $[c_{ij}]$:

|      | P1   | P2   | P3   | P4   | P5   | P6   |
|------|------|------|------|------|------|------|
| MA   | 2216 | 1911 | 1661 | 2122 | 1442 | 1442 |
| MB   | 1100 | 1271 | 2764 | 2557 | 1036 | 1036 |
| MC   | 2827 | 2784 | 2206 | 2216 | 2677 | 2677 |
| MD   | 2627 | 1273 | 2610 | 1957 | 1594 | 1594 |
| ME   | 3359 | 1003 | 2554 | 1706 | 2065 | 2065 |
| MF   | 1579 | 2289 | 2368 | 1922 | 2740 | 2740 |

##### Summary

Assign each manager to exactly one project and each project to exactly one manager, minimizing the total assignment cost as given in the table above.