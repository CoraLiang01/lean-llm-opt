##### Decision Variables

$x_{ij} \in \{0,1\}$: $x_{ij} = 1$ if machine $i$ is assigned to task $j$, $0$ otherwise, for all $i \in M$, $j \in T$.

##### Parameters

Let $M = \{\text{M1}, \text{M2}, \ldots, \text{M12}\}$ (machines), $T = \{\text{A}, \text{B}, \ldots, \text{L}\}$ (tasks).

Let $c_{ij}$ be the machining cost of assigning machine $i$ to task $j$, as given below:

\[
\begin{array}{c|cccccccccccc}
 & \text{A} & \text{B} & \text{C} & \text{D} & \text{E} & \text{F} & \text{G} & \text{H} & \text{I} & \text{J} & \text{K} & \text{L} \\
\hline
\text{M1} & 167.4 & 98.6 & 189.4 & 119.6 & 182.0 & 145.1 & 185.4 & 94.8 & 122.3 & 123.3 & 96.1 & 90.3 \\
\text{M2} & 156.2 & 88.7 & 187.3 & 124.7 & 173.2 & 144.3 & 179.0 & 91.5 & 115.1 & 119.5 & 100.1 & 88.6 \\
\text{M3} & 184.3 & 121.0 & 216.6 & 140.0 & 196.2 & 168.8 & 205.6 & 114.2 & 133.3 & 144.5 & 116.0 & 107.7 \\
\text{M4} & 157.9 & 92.9 & 185.1 & 120.3 & 175.1 & 146.2 & 180.8 & 86.3 & 111.6 & 115.9 & 98.1 & 91.1 \\
\text{M5} & 175.6 & 103.6 & 204.5 & 130.0 & 192.8 & 157.5 & 194.2 & 106.9 & 129.9 & 134.9 & 105.8 & 98.6 \\
\text{M6} & 166.8 & 107.0 & 199.2 & 130.4 & 183.6 & 159.5 & 187.0 & 98.2 & 121.3 & 126.2 & 105.9 & 101.8 \\
\text{M7} & 159.7 & 93.2 & 183.8 & 113.0 & 171.9 & 139.1 & 169.6 & 85.1 & 110.0 & 116.7 & 90.6 & 85.2 \\
\text{M8} & 184.8 & 115.9 & 205.1 & 138.6 & 195.4 & 160.1 & 200.2 & 108.5 & 136.9 & 140.0 & 114.6 & 103.9 \\
\text{M9} & 157.3 & 86.2 & 186.0 & 113.9 & 166.2 & 136.8 & 167.5 & 78.8 & 107.4 & 114.5 & 87.2 & 78.6 \\
\text{M10} & 164.8 & 97.8 & 200.9 & 125.8 & 188.9 & 151.2 & 187.7 & 99.5 & 119.5 & 132.1 & 101.1 & 98.4 \\
\text{M11} & 164.0 & 92.2 & 186.2 & 115.7 & 174.5 & 143.0 & 175.9 & 92.3 & 114.0 & 121.2 & 93.7 & 91.2 \\
\text{M12} & 151.7 & 76.7 & 179.5 & 109.5 & 160.6 & 128.4 & 170.2 & 74.4 & 103.7 & 110.4 & 83.7 & 75.2 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in M} \sum_{j \in T} c_{ij} x_{ij}
\]

##### Constraints

1. Each machine is assigned to exactly one task:
   \[
   \sum_{j \in T} x_{ij} = 1 \quad \forall i \in M
   \]
2. Each task is assigned to exactly one machine:
   \[
   \sum_{i \in M} x_{ij} = 1 \quad \forall j \in T
   \]
3. Binary variables:
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in T
   \]