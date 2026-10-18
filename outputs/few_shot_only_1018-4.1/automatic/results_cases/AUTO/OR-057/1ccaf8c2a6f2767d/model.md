Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i \in \{1,2,\ldots,15\}$ (PlatformID from capacity.csv)
- $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (ProductName from products.csv)

**Parameters:**
- $v_j$: Value of genre $j$ (from Value column)
- $w_j$: Memory requirement of genre $j$ (from Weight column)
- $C_i$: Memory capacity of platform $i$ (from Capacity column)

**Data:**

Platforms and capacities:
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

Games and their values and memory requirements:
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

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{15} \sum_{j \in \mathcal{J}} v_j x_{ij}
\]
where $\mathcal{J}$ is the set of all genres listed above.

**Subject to:**

For each platform $i$:
\[
\sum_{j \in \mathcal{J}} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,15
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,15,\; j \in \mathcal{J}
\]

**Where:**

- $v_j$ and $w_j$ are as given in the table above for each genre $j$.
- $C_i$ is as given in the table above for each platform $i$.

**Decision variables:**
- $x_{ij}$: integer, $\geq 0$ (number of units of genre $j$ to list on platform $i$)