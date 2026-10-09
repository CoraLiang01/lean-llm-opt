Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let $P$ be the set of platforms, indexed by PlatformID:
$$
P = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}
$$

Let $G$ be the set of game genres (products), indexed by their names:
$$
G = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
$$

Let $v_j$ be the value of genre $j$ and $w_j$ its memory requirement (Weight), as given below:

| Genre        | $v_j$ | $w_j$ |
|--------------|-------|-------|
| Racing       | 59    | 776   |
| Sports       | 83    | 573   |
| Action       | 94    | 127   |
| Adventure    | 41    | 138   |
| RPG          | 96    | 385   |
| Shooter      | 12    | 263   |
| Strategy     | 83    | 473   |
| Simulation   | 36    | 387   |
| Puzzle       | 56    | 390   |
| Fighting     | 27    | 556   |
| Platformer   | 47    | 601   |
| Survival     | 24    | 441   |
| Horror       | 14    | 603   |
| Sandbox      | 22    | 411   |
| MMO          | 17    | 652   |

Let $C_i$ be the memory capacity of platform $i$:

| PlatformID | $C_i$ |
|------------|-------|
| 1          | 995   |
| 2          | 1143  |
| 3          | 949   |
| 4          | 969   |
| 5          | 1649  |
| 6          | 870   |
| 7          | 1064  |
| 8          | 536   |
| 9          | 766   |
| 10         | 532   |
| 11         | 1703  |
| 12         | 1633  |
| 13         | 1203  |
| 14         | 1979  |
| 15         | 1797  |

The complete mathematical model is:

$$
\textbf{Objective:} \\
\max \sum_{i \in P} \sum_{j \in G} v_j \cdot x_{ij}
$$

$$
\textbf{Subject to:}
$$

$$
\sum_{j \in G} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in P
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in P,\, j \in G
$$

Where:
- $x_{ij}$: number of units of game genre $j$ to be listed on platform $i$ (integer, $\geq 0$)
- $v_j$: value of genre $j$ (see table above)
- $w_j$: memory requirement of genre $j$ (see table above)
- $C_i$: memory capacity of platform $i$ (see table above)

All coefficients and identifiers are as retrieved and preserved in source order.