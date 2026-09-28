##### Decision Variables

$x_{ij} \in \{0,1\}$:  
$x_{ij} = 1$ if manager $i$ is assigned to project $j$, $0$ otherwise, for all managers $i \in I$ and projects $j \in J$.

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

where $c_{ij}$ is the cost of assigning manager $i$ to project $j$ (see cost matrix below).

##### Constraints

1. Each manager is assigned to exactly one project:  
   $\sum_{j \in J} x_{ij} = 1, \quad \forall i \in I$

2. Each project is assigned to exactly one manager:  
   $\sum_{i \in I} x_{ij} = 1, \quad \forall j \in J$

3. Binary assignment variables:  
   $x_{ij} \in \{0,1\}, \quad \forall i \in I, \forall j \in J$

##### Sets

$I = \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$ (Managers)  
$J = \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$ (Projects)

##### Cost Matrix

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

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{j \in J} x_{ij} = 1, \quad \forall i \in I \\
& \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J \\
& x_{ij} \in \{0,1\}, \quad \forall i \in I, \forall j \in J \\
\end{align*}
\]

where $c_{ij}$ is given by the cost matrix above.