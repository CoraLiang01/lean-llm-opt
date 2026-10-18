##### Decision Variables

$x_{ij}\in\{0,1\}$: assignment of manager $i\in I$ to project $j\in J$.

##### Objective Function

$\displaystyle \min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}$

##### Constraints

1. Manager assignment: $\displaystyle \sum_{j\in J}x_{ij}=1,\quad \forall i\in I$  
2. Project assignment: $\displaystyle \sum_{i\in I}x_{ij}=1,\quad \forall j\in J$  
3. Domains: $x_{ij}\in\{0,1\}$.

##### Parameters

$I=\{MA,MB,MC,MD,ME,MF\}$, $J=\{P1,P2,P3,P4,P5,P6\}$  

$C=\begin{bmatrix}
2216 & 1911 & 1661 & 2122 & 1442 & 1442\\
1100 & 1271 & 2764 & 2557 & 1036 & 1036\\
2827 & 2784 & 2206 & 2216 & 2677 & 2677\\
2627 & 1273 & 2610 & 1957 & 1594 & 1594\\
3359 & 1003 & 2554 & 1706 & 2065 & 2065\\
1579 & 2289 & 2368 & 1922 & 2740 & 2740
\end{bmatrix}$  

where $c_{ij}$ is the entry in row $i$ and column $j$ of $C$.