Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Indices:
- $i$ indexes PlatformID $\in \{1,2,\ldots,15\}$
- $j$ indexes ProductName (game genres) in the order given below

Parameters (from the data):

Platforms and their capacities:
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
\text{ProductName} & \text{Value} & \text{Weight (Memory Requirement)} \\
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
\max \sum_{i=1}^{15} \sum_{j=1}^{15} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of genre $j$ as listed above.

Subject to, for each platform $i$:
\[
\sum_{j=1}^{15} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,15
\]
where $w_j$ is the Weight (memory requirement) of genre $j$, and $C_i$ is the Capacity of platform $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,15;\ j = 1,\ldots,15
\]

All identifiers and coefficients are as given in the data above, in original order.