##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of game $j$ (genre) to be listed on platform $i$.

Let $i \in I = \{1,2,\ldots,15\}$ (platforms), $j \in J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ (games/genres).

##### Parameters

- Platform capacities $C_i$:

| $i$ | $C_i$ |
|----|------|
| 1  | 995  |
| 2  | 1143 |
| 3  | 949  |
| 4  | 969  |
| 5  | 1649 |
| 6  | 870  |
| 7  | 1064 |
| 8  | 536  |
| 9  | 766  |
| 10 | 532  |
| 11 | 1703 |
| 12 | 1633 |
| 13 | 1203 |
| 14 | 1979 |
| 15 | 1797 |

- Game values $v_j$ and weights $w_j$:

| $j$           | $v_j$ | $w_j$ |
|---------------|-------|-------|
| Racing        | 59    | 776   |
| Sports        | 83    | 573   |
| Action        | 94    | 127   |
| Adventure     | 41    | 138   |
| RPG           | 96    | 385   |
| Shooter       | 12    | 263   |
| Strategy      | 83    | 473   |
| Simulation    | 36    | 387   |
| Puzzle        | 56    | 390   |
| Fighting      | 27    | 556   |
| Platformer    | 47    | 601   |
| Survival      | 24    | 441   |
| Horror        | 14    | 603   |
| Sandbox       | 22    | 411   |
| MMO           | 17    | 652   |

##### Objective Function

$$
\max \sum_{i=1}^{15} \sum_{j \in J} v_j x_{ij}
$$

##### Constraints

For each platform $i=1,\ldots,15$:
$$
\sum_{j \in J} w_j x_{ij} \leq C_i
$$

For all $i=1,\ldots,15$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

##### Complete Model

$$
\begin{align*}
\max\quad & \sum_{i=1}^{15} \sum_{j \in J} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j \in J} w_j x_{ij} \leq C_i \quad \forall i=1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,15,\ j \in J
\end{align*}
$$

Where all coefficients and identifiers are as listed above.