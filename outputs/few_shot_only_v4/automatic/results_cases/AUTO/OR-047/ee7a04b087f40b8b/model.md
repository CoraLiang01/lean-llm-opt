Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are nonnegative integers.

Let:
- $i$ index platforms, with PlatformId as in capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$ index genres, with ProductName as in products.csv: $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$
- $v_j$ is the Value of genre $j$ (from products.csv)
- $w_j$ is the Weight (memory requirement) of genre $j$ (from products.csv)
- $C_i$ is the Capacity of platform $i$ (from capacity.csv)

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
\]

Subject to, for each platform $i$:
\[
\sum_{j=1}^{15} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:

Platform Capacities:
\[
\begin{array}{ll}
C_1 = 1336 & C_6 = 627 \\
C_2 = 1754 & C_7 = 748 \\
C_3 = 1617 & C_8 = 1540 \\
C_4 = 1119 & C_9 = 1292 \\
C_5 = 1410 & C_{10} = 1138 \\
\end{array}
\]

Genre Values and Weights:
\[
\begin{array}{lll}
\text{Genre} & v_j & w_j \\
\hline
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

Full Model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t. } & \sum_{j=1}^{15} w_j x_{ij} \leq C_i, \quad \forall i \in \{1,\ldots,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i, j
\end{align*}
\]

With all $v_j$, $w_j$, and $C_i$ as listed above.