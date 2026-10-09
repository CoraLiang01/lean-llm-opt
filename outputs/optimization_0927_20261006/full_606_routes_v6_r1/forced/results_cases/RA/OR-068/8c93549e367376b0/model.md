Let $x_{ij}$ be a binary variable equal to 1 if manager $i$ is assigned to project $j$, and 0 otherwise.

**Sets:**
- Managers: $\{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$
- Projects: $\{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

**Parameters:**
- $c_{ij}$: Cost of assigning manager $i$ to project $j$, as given below.

|        | P1   | P2   | P3   | P4   | P5   | P6   |
|--------|------|------|------|------|------|------|
| MA     | 2216 | 1911 | 1661 | 2122 | 1442 | 1442 |
| MB     | 1100 | 1271 | 2764 | 2557 | 1036 | 1036 |
| MC     | 2827 | 2784 | 2206 | 2216 | 2677 | 2677 |
| MD     | 2627 | 1273 | 2610 | 1957 | 1594 | 1594 |
| ME     | 3359 | 1003 | 2554 | 1706 | 2065 | 2065 |
| MF     | 1579 | 2289 | 2368 | 1922 | 2740 | 2740 |

**Variables:**
- $x_{ij} \in \{0,1\}$ for all managers $i$ and projects $j$.

**Objective:**
\[
\min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} c_{ij} x_{ij}
\]

**Subject to:**

1. **Each manager is assigned to exactly one project:**
   \[
   \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}
   \]

2. **Each project is assigned to exactly one manager:**
   \[
   \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}
   \]

3. **Binary assignment variables:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i, j
   \]

**Parameter Table:**

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

**Variable Domains:**
\[
x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}
\]