##### Decision Variables

Let $x_{ij} \in \{0,1\}$ be a binary variable, where $x_{ij} = 1$ if manager $i$ is assigned to project $j$, and $0$ otherwise.

Managers: $I = \{\text{MA}, \text{MB}, \text{MC}\}$
Projects: $J = \{\text{P1}, \text{P2}, \text{P3}\}$

##### Parameters

The assignment cost matrix $c_{ij}$ is:

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

Managers: $\{\text{MA}, \text{MB}, \text{MC}\}$

Projects: $\{\text{P1}, \text{P2}, \text{P3}\}$

Assignment cost matrix $c_{ij}$:
- MA: P1 = 3000, P2 = 3200, P3 = 3100
- MB: P1 = 2800, P2 = 3300, P3 = 2900
- MC: P1 = 2900, P2 = 3100, P3 = 3000