Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$. The variables $x_{ij}$ are nonnegative integers.

Indices:
- $i$ indexes platforms: $i \in \{1,2,\ldots,15\}$ (PlatformID from capacity.csv)
- $j$ indexes game genres: $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ (ProductName from products.csv)

Parameters:
- $v_j$: Value of one unit of genre $j$ (from Value column)
- $w_j$: Memory requirement of one unit of genre $j$ (from Weight column)
- $C_i$: Memory capacity of platform $i$ (from Capacity column)

Data:

Platforms and Capacities:
\[
\begin{array}{ll}
\text{PlatformID} & C_i \\
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

Game Genres, Values, and Weights:
\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
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

Model:

Objective:
\[
\max \sum_{i=1}^{15} \sum_{j \in \mathcal{J}} v_j x_{ij}
\]
where $\mathcal{J}$ is the set of all genres listed above.

Subject to:

Platform memory capacity constraints:
\[
\sum_{j \in \mathcal{J}} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,15\},\; j \in \mathcal{J}
\]

Where all coefficients and identifiers are as given in the tables above.