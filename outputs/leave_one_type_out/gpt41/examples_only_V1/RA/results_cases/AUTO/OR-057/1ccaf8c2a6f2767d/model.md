Sets and Indices:
- Let I be the set of platforms, indexed by i. (PlatformID from capacity.csv)
- Let J be the set of games, indexed by j. (ProductName from products.csv)

Parameters:
- Capacity_i: Memory capacity of platform i (from capacity.csv, column Capacity)
- Value_j: Value of game j (from products.csv, column Value)
- Weight_j: Memory requirement of game j (from products.csv, column Weight)

Decision Variables:
- x_{i,j}: Number of units of game j to be listed on platform i (integer, x_{i,j} ≥ 0)

Mathematical Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i \in I} \sum_{j \in J} \text{Value}_j \cdot x_{i,j}
\]

Subject to platform memory capacities:
\[
\sum_{j \in J} \text{Weight}_j \cdot x_{i,j} \leq \text{Capacity}_i \quad \forall i \in I
\]

Variable domains:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

Where:

From capacity.csv:
\[
\begin{array}{ll}
\text{PlatformID} & \text{Capacity} \\
1 & 995 \\
2 & 1143 \\
3 & 949 \\
4 & 969 \\
5 & 1649 \\
6 & 870 \\
7 & 1064 \\
8 & 536 \\
9 & 766 \\
10 & 532 \\
11 & 1703 \\
12 & 1633 \\
13 & 1203 \\
14 & 1979 \\
15 & 1797 \\
\end{array}
\]

From products.csv:
\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
\text{Racing} & 59 & 776 \\
\text{Sports} & 83 & 573 \\
\text{Action} & 94 & 127 \\
\text{Adventure} & 41 & 138 \\
\text{RPG} & 96 & 385 \\
\text{Shooter} & 12 & 263 \\
\text{Strategy} & 83 & 473 \\
\text{Simulation} & 36 & 387 \\
\text{Puzzle} & 56 & 390 \\
\text{Fighting} & 27 & 556 \\
\text{Platformer} & 47 & 601 \\
\text{Survival} & 24 & 441 \\
\text{Horror} & 14 & 603 \\
\text{Sandbox} & 22 & 411 \\
\text{MMO} & 17 & 652 \\
\end{array}
\]

Explicitly, for each platform i (PlatformID 1 to 15), and each game j (ProductName as above), the model is:

Maximize
\[
\sum_{i=1}^{15} \sum_{j \in J} \text{Value}_j \cdot x_{i,j}
\]

Subject to, for each i:
\[
\sum_{j \in J} \text{Weight}_j \cdot x_{i,j} \leq \text{Capacity}_i
\]

and
\[
x_{i,j} \in \mathbb{Z}_{\geq 0}
\]

where all coefficients and sets are as listed above.