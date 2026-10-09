##### Sets and Indices

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of platforms, indexed by $i$.

Let $J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ be the set of game genres, indexed by $j$.

##### Parameters

Platform capacities:
\[
\begin{align*}
C_1 &= 1336 \\
C_2 &= 1754 \\
C_3 &= 1617 \\
C_4 &= 1119 \\
C_5 &= 1410 \\
C_6 &= 627 \\
C_7 &= 748 \\
C_8 &= 1540 \\
C_9 &= 1292 \\
C_{10} &= 1138 \\
\end{align*}
\]

Game genre values and memory requirements:
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

##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of games from genre $j$ to be listed on platform $i$.

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Platform memory capacity:**
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

###### Retrieved Information

- Platforms and capacities:
  - 1: 1336
  - 2: 1754
  - 3: 1617
  - 4: 1119
  - 5: 1410
  - 6: 627
  - 7: 748
  - 8: 1540
  - 9: 1292
  - 10: 1138

- Genres, values, and weights:
  - Racing: value 28, weight 393
  - Sports: value 69, weight 195
  - Action: value 20, weight 192
  - Adventure: value 62, weight 155
  - RPG: value 58, weight 500
  - Shooter: value 11, weight 156
  - Strategy: value 73, weight 317
  - Simulation: value 43, weight 694
  - Puzzle: value 28, weight 751
  - Fighting: value 57, weight 467
  - Platformer: value 92, weight 796
  - Survival: value 66, weight 146
  - Horror: value 14, weight 269
  - Sandbox: value 49, weight 246
  - MMO: value 12, weight 652