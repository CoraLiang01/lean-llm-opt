##### Sets and Indices

- Let $I = \{1,2,\ldots,15\}$ be the set of platforms, indexed by $i$.
- Let $J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ be the set of games (genres), indexed by $j$.

##### Parameters

- Platform capacities:
  - $C_1 = 995$
  - $C_2 = 1143$
  - $C_3 = 949$
  - $C_4 = 969$
  - $C_5 = 1649$
  - $C_6 = 870$
  - $C_7 = 1064$
  - $C_8 = 536$
  - $C_9 = 766$
  - $C_{10} = 532$
  - $C_{11} = 1703$
  - $C_{12} = 1633$
  - $C_{13} = 1203$
  - $C_{14} = 1979$
  - $C_{15} = 1797$

- Game values and memory requirements:

| $j$           | Value $v_j$ | Weight $w_j$ |
|---------------|-------------|--------------|
| Racing        | 59          | 776          |
| Sports        | 83          | 573          |
| Action        | 94          | 127          |
| Adventure     | 41          | 138          |
| RPG           | 96          | 385          |
| Shooter       | 12          | 263          |
| Strategy      | 83          | 473          |
| Simulation    | 36          | 387          |
| Puzzle        | 56          | 390          |
| Fighting      | 27          | 556          |
| Platformer    | 47          | 601          |
| Survival      | 24          | 441          |
| Horror        | 14          | 603          |
| Sandbox       | 22          | 411          |
| MMO           | 17          | 652          |

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of game $j$ to be listed on platform $i$ (integer, nonnegative).

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Platform memory capacity:**
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

##### Parameters (full listing)

- $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$
- $C = [995, 1143, 949, 969, 1649, 870, 1064, 536, 766, 532, 1703, 1633, 1203, 1979, 1797]$
- $v = [59, 83, 94, 41, 96, 12, 83, 36, 56, 27, 47, 24, 14, 22, 17]$
- $w = [776, 573, 127, 138, 385, 263, 473, 387, 390, 556, 601, 441, 603, 411, 652]$

##### Complete Mathematical Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^{15} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{15} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,15;\ j=1,\ldots,15
\end{align*}
\]

where $v_j$ and $w_j$ are as listed above for each game $j$, and $C_i$ is the capacity for each platform $i$ as listed above.