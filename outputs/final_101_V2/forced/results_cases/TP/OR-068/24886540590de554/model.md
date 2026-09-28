##### Decision Variables

Let $x_{ij} \in \{0,1\}$ for all managers $i \in I$ and projects $j \in J$, where $x_{ij} = 1$ if manager $i$ is assigned to project $j$, and $0$ otherwise.

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

##### Complete Model

\[
\begin{align*}
\min\ & 2216x_{\text{MA},\text{P1}} + 1911x_{\text{MA},\text{P2}} + 1661x_{\text{MA},\text{P3}} + 2122x_{\text{MA},\text{P4}} + 1442x_{\text{MA},\text{P5}} + 1442x_{\text{MA},\text{P6}} \\
&+ 1100x_{\text{MB},\text{P1}} + 1271x_{\text{MB},\text{P2}} + 2764x_{\text{MB},\text{P3}} + 2557x_{\text{MB},\text{P4}} + 1036x_{\text{MB},\text{P5}} + 1036x_{\text{MB},\text{P6}} \\
&+ 2827x_{\text{MC},\text{P1}} + 2784x_{\text{MC},\text{P2}} + 2206x_{\text{MC},\text{P3}} + 2216x_{\text{MC},\text{P4}} + 2677x_{\text{MC},\text{P5}} + 2677x_{\text{MC},\text{P6}} \\
&+ 2627x_{\text{MD},\text{P1}} + 1273x_{\text{MD},\text{P2}} + 2610x_{\text{MD},\text{P3}} + 1957x_{\text{MD},\text{P4}} + 1594x_{\text{MD},\text{P5}} + 1594x_{\text{MD},\text{P6}} \\
&+ 3359x_{\text{ME},\text{P1}} + 1003x_{\text{ME},\text{P2}} + 2554x_{\text{ME},\text{P3}} + 1706x_{\text{ME},\text{P4}} + 2065x_{\text{ME},\text{P5}} + 2065x_{\text{ME},\text{P6}} \\
&+ 1579x_{\text{MF},\text{P1}} + 2289x_{\text{MF},\text{P2}} + 2368x_{\text{MF},\text{P3}} + 1922x_{\text{MF},\text{P4}} + 2740x_{\text{MF},\text{P5}} + 2740x_{\text{MF},\text{P6}} \\
\text{s.t.} \\
&\sum_{j \in J} x_{ij} = 1 \quad \forall i \in I \\
&\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J \\
&x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
\end{align*}
\]