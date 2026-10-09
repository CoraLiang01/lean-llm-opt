##### Decision Variables

Let $x_{ij} \in \{0,1\}$ for all managers $i$ and projects $j$, where $x_{ij}=1$ if manager $i$ is assigned to project $j$, and $0$ otherwise.

##### Parameters

Managers: $I = \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$

Projects: $J = \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

Assignment costs $c_{ij}$:

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

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. Each manager is assigned to exactly one project:
   \[
   \sum_{j \in J} x_{ij} = 1 \quad \forall i \in I
   \]

2. Each project is assigned to exactly one manager:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

3. Binary assignment variables:
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]

##### Retrieved Information

{
  "managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "projects": ["P1", "P2", "P3", "P4", "P5", "P6"],
  "cost": {
    "MA": {"P1": 2216, "P2": 1911, "P3": 1661, "P4": 2122, "P5": 1442, "P6": 1442},
    "MB": {"P1": 1100, "P2": 1271, "P3": 2764, "P4": 2557, "P5": 1036, "P6": 1036},
    "MC": {"P1": 2827, "P2": 2784, "P3": 2206, "P4": 2216, "P5": 2677, "P6": 2677},
    "MD": {"P1": 2627, "P2": 1273, "P3": 2610, "P4": 1957, "P5": 1594, "P6": 1594},
    "ME": {"P1": 3359, "P2": 1003, "P3": 2554, "P4": 1706, "P5": 2065, "P6": 2065},
    "MF": {"P1": 1579, "P2": 2289, "P3": 2368, "P4": 1922, "P5": 2740, "P6": 2740}
  }
}