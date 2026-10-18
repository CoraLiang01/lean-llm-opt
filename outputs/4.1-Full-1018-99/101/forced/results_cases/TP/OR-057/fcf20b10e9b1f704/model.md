##### Sets

Let $I = \{1,2,\ldots,15\}$ be the set of platforms (PlatformID from capacity.csv, in source order).

Let $J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ be the set of games (ProductName from products.csv, in source order).

##### Parameters

For each platform $i \in I$:

- $C_i$ = platform $i$'s memory capacity

\[
\begin{array}{ll}
C_1 = 995 & C_9 = 766 \\
C_2 = 1143 & C_{10} = 532 \\
C_3 = 949 & C_{11} = 1703 \\
C_4 = 969 & C_{12} = 1633 \\
C_5 = 1649 & C_{13} = 1203 \\
C_6 = 870 & C_{14} = 1979 \\
C_7 = 1064 & C_{15} = 1797 \\
C_8 = 536 & \\
\end{array}
\]

For each game $j \in J$:

- $v_j$ = value of game $j$
- $w_j$ = memory requirement of game $j$

\[
\begin{array}{lll}
\text{Game} & v_j & w_j \\
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

##### Decision Variables

For each platform $i \in I$ and game $j \in J$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of game $j$ to list on platform $i$

##### Objective

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

For each platform $i \in I$:

\[
\sum_{j \in J} w_j x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

##### Retrieved Information

Platforms (in source order):

- 1: 995
- 2: 1143
- 3: 949
- 4: 969
- 5: 1649
- 6: 870
- 7: 1064
- 8: 536
- 9: 766
- 10: 532
- 11: 1703
- 12: 1633
- 13: 1203
- 14: 1979
- 15: 1797

Games (in source order):

- Racing: value 59, weight 776
- Sports: value 83, weight 573
- Action: value 94, weight 127
- Adventure: value 41, weight 138
- RPG: value 96, weight 385
- Shooter: value 12, weight 263
- Strategy: value 83, weight 473
- Simulation: value 36, weight 387
- Puzzle: value 56, weight 390
- Fighting: value 27, weight 556
- Platformer: value 47, weight 601
- Survival: value 24, weight 441
- Horror: value 14, weight 603
- Sandbox: value 22, weight 411
- MMO: value 17, weight 652