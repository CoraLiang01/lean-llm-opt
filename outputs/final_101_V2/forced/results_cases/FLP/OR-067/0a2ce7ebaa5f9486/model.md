##### Decision Variables

Let $x_{ij} \in \{0,1\}$ indicate whether manager $i$ is assigned to project $j$ ($x_{ij}=1$ if manager $i$ is assigned to project $j$, $0$ otherwise), for all managers $i \in I$ and projects $j \in J$.

##### Parameters

- $I = \{\text{MA}, \text{MB}, \text{MC}\}$ (set of managers)
- $J = \{\text{P1}, \text{P2}, \text{P3}\}$ (set of projects)
- Cost matrix $c_{ij}$:

\[
\begin{array}{c|ccc}
 & \text{P1} & \text{P2} & \text{P3} \\
\hline
\text{MA} & 3000 & 3200 & 3100 \\
\text{MB} & 2800 & 3300 & 2900 \\
\text{MC} & 2900 & 3100 & 3000 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. Each manager is assigned to exactly one project:
   \[
   \sum_{j \in J} x_{ij} = 1, \quad \forall i \in I
   \]

2. Each project is assigned to exactly one manager:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

3. Binary assignment variables:
   \[
   x_{ij} \in \{0,1\}, \quad \forall i \in I,\, j \in J
   \]

##### Retrieved Information

- Managers: MA, MB, MC
- Projects: P1, P2, P3
- Cost matrix $c_{ij}$:
  - MA: P1 = 3000, P2 = 3200, P3 = 3100
  - MB: P1 = 2800, P2 = 3300, P3 = 2900
  - MC: P1 = 2900, P2 = 3100, P3 = 3000

This model assigns each manager to exactly one project and each project to exactly one manager, minimizing the total assignment cost.