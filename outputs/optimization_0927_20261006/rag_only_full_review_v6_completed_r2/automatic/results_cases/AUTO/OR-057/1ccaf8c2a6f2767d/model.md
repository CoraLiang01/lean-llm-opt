Let:
- I = set of platforms, indexed by i (PlatformID from capacity.csv, in order: 1,2,...,15)
- J = set of games, indexed by j (ProductName from products.csv, in order: Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO)
- Capacity_i = memory capacity of platform i (from capacity.csv)
- Value_j = value of game j (from products.csv)
- Weight_j = memory requirement of game j (from products.csv)
- x_ij = integer variable: number of units of game j to list on platform i (x_ij ≥ 0, integer)

Model:

Maximize
\[
\sum_{i \in I} \sum_{j \in J} Value_j \cdot x_{ij}
\]
That is,
\[
\text{Maximize } 
\sum_{i=1}^{15} \sum_{j=1}^{15} Value_j \cdot x_{ij}
\]
where the Value_j are as follows (in order):
\[
\begin{array}{ll}
\text{Racing:} & 59 \\
\text{Sports:} & 83 \\
\text{Action:} & 94 \\
\text{Adventure:} & 41 \\
\text{RPG:} & 96 \\
\text{Shooter:} & 12 \\
\text{Strategy:} & 83 \\
\text{Simulation:} & 36 \\
\text{Puzzle:} & 56 \\
\text{Fighting:} & 27 \\
\text{Platformer:} & 47 \\
\text{Survival:} & 24 \\
\text{Horror:} & 14 \\
\text{Sandbox:} & 22 \\
\text{MMO:} & 17 \\
\end{array}
\]

Subject to, for each platform i (in the order of PlatformID 1 to 15):

\[
\sum_{j \in J} Weight_j \cdot x_{ij} \leq Capacity_i
\]
where the Weight_j are as follows (in order):
\[
\begin{array}{ll}
\text{Racing:} & 776 \\
\text{Sports:} & 573 \\
\text{Action:} & 127 \\
\text{Adventure:} & 138 \\
\text{RPG:} & 385 \\
\text{Shooter:} & 263 \\
\text{Strategy:} & 473 \\
\text{Simulation:} & 387 \\
\text{Puzzle:} & 390 \\
\text{Fighting:} & 556 \\
\text{Platformer:} & 601 \\
\text{Survival:} & 441 \\
\text{Horror:} & 603 \\
\text{Sandbox:} & 411 \\
\text{MMO:} & 652 \\
\end{array}
\]

And the platform capacities (in order):

\[
\begin{array}{ll}
\text{Platform 1:} & 995 \\
\text{Platform 2:} & 1143 \\
\text{Platform 3:} & 949 \\
\text{Platform 4:} & 969 \\
\text{Platform 5:} & 1649 \\
\text{Platform 6:} & 870 \\
\text{Platform 7:} & 1064 \\
\text{Platform 8:} & 536 \\
\text{Platform 9:} & 766 \\
\text{Platform 10:} & 532 \\
\text{Platform 11:} & 1703 \\
\text{Platform 12:} & 1633 \\
\text{Platform 13:} & 1203 \\
\text{Platform 14:} & 1979 \\
\text{Platform 15:} & 1797 \\
\end{array}
\]

So, for each i = 1,...,15:
\[
\sum_{j=1}^{15} Weight_j \cdot x_{ij} \leq Capacity_i
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

All data is used in the original order as required. This is a complete integer programming formulation for the described problem.