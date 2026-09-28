Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$, where $i \in \{1,2,\ldots,10\}$ (PlatformId from capacity.csv) and $j$ indexes the genres (ProductName from products.csv). All $x_{ij}$ are integer and $\geq 0$.

Parameters:

- $v_j$: Value of one unit of genre $j$ (from Value column in products.csv)
- $w_j$: Memory requirement of one unit of genre $j$ (from Weight column in products.csv)
- $C_i$: Memory capacity of platform $i$ (from Capacity column in capacity.csv)

Sets:

- Platforms $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (PlatformId)
- Genres $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (ProductName)

Model:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j} v_j x_{ij}
\]

Subject to, for each platform $i$:
\[
\sum_{j} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Numerical Data:

Platform Capacities (from capacity.csv, in source order):

\[
\begin{array}{ll}
\text{PlatformId} & C_i \\
1 & 1336 \\
2 & 1754 \\
3 & 1617 \\
4 & 1119 \\
5 & 1410 \\
6 & 627 \\
7 & 748 \\
8 & 1540 \\
9 & 1292 \\
10 & 1138 \\
\end{array}
\]

Game Genres, Values, and Memory Requirements (from products.csv, in source order):

\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
\text{Racing} & 28 & 393 \\
\text{Sports} & 69 & 195 \\
\text{Action} & 20 & 192 \\
\text{Adventure} & 62 & 155 \\
\text{RPG} & 58 & 500 \\
\text{Shooter} & 11 & 156 \\
\text{Strategy} & 73 & 317 \\
\text{Simulation} & 43 & 694 \\
\text{Puzzle} & 28 & 751 \\
\text{Fighting} & 57 & 467 \\
\text{Platformer} & 92 & 796 \\
\text{Survival} & 66 & 146 \\
\text{Horror} & 14 & 269 \\
\text{Sandbox} & 49 & 246 \\
\text{MMO} & 12 & 652 \\
\end{array}
\]

Decision variables:

\[
x_{ij} = \text{number of units of games from genre } j \text{ to be listed on platform } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Complete Model:

\[
\boxed{
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
}
\]

Where all $v_j$, $w_j$, and $C_i$ are as listed above, and $i$ and $j$ are indexed in the original source order.