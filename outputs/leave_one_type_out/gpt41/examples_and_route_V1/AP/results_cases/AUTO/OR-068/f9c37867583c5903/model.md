Let $M = \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$ be the set of managers, and $P = \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$ be the set of projects.

Let $c_{ij}$ denote the cost of assigning manager $i$ to project $j$, as given in the table below.

Let $x_{ij}$ be a binary variable equal to 1 if manager $i$ is assigned to project $j$, and 0 otherwise.

Minimize total assignment cost:
$$
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
$$

Subject to:

Each manager is assigned to exactly one project:
$$
\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M
$$

Each project is assigned to exactly one manager:
$$
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P
$$

Binary assignment variables:
$$
x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in P
$$

Where the cost coefficients $c_{ij}$ are:

\[
\begin{array}{c|cccccc}
 & \text{P1} & \text{P2} & \text{P3} & \text{P4} & \text{P5} & \text{P6} \\
\hline
\text{MA} & 2216 & 1911 & 1661 & 2122 & 1442 & 1442 \\
\text{MB} & 1100 & 1271 & 2764 & 2557 & 1036 & 1036 \\
\text{MC} & 2827 & 2784 & 2206 & 2216 & 2677 & 2677 \\
\text{MD} & 2627 & 1273 & 2610 & 1957 & 1594 & 1594 \\
\text{ME} & 3359 & 1003 & 2554 & 1706 & 2065 & 2065 \\
\text{MF} & 1579 & 2289 & 2368 & 1922 & 2740 & 2740 \\
\end{array}
\]

All variables and constraints use the original identifiers and coefficients as retrieved.