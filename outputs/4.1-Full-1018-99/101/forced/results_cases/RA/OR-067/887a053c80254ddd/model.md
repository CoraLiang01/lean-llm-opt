Let $x_{ij}$ be a binary variable equal to 1 if manager $i$ is assigned to project $j$, and 0 otherwise.

Let $M = \{\text{MA}, \text{MB}, \text{MC}\}$ (managers) and $P = \{\text{P1}, \text{P2}, \text{P3}\}$ (projects).

Let $c_{ij}$ be the cost for manager $i$ to complete project $j$, as given below:

\[
\begin{array}{c|ccc}
 & \text{P1} & \text{P2} & \text{P3} \\
\hline
\text{MA} & 3000 & 3200 & 3100 \\
\text{MB} & 2800 & 3300 & 2900 \\
\text{MC} & 2900 & 3100 & 3000 \\
\end{array}
\]

Objective:
\[
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]

Subject to:

Each manager is assigned to exactly one project:
\[
\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M
\]

Each project is assigned to exactly one manager:
\[
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P
\]

Binary assignment variables:
\[
x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in P
\]

Where:
- $M = \{\text{MA}, \text{MB}, \text{MC}\}$
- $P = \{\text{P1}, \text{P2}, \text{P3}\}$
- $c_{ij}$ as given in the table above.