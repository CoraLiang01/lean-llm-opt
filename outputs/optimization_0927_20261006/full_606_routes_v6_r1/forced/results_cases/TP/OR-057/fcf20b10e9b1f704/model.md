##### Sets

Let $I = \{1,2,\ldots,15\}$ be the set of platforms (PlatformID from capacity.csv).  
Let $J = \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ be the set of games (ProductName from products.csv).

##### Parameters

For each platform $i \in I$:
- $C_i$ = platform $i$'s memory capacity

For each game $j \in J$:
- $v_j$ = value of game $j$
- $w_j$ = memory requirement (weight) of game $j$

##### Decision Variables

For each $i \in I$, $j \in J$:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of game $j$ to be listed on platform $i$

##### Objective

$\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

##### Constraints

For each platform $i \in I$:
- $\sum_{j \in J} w_j x_{ij} \leq C_i$

For all $i \in I$, $j \in J$:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

##### Numerical Data

**Platforms and Capacities (from capacity.csv, in source order):**

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

**Games, Values, and Weights (from products.csv, in source order):**

| ProductName  | $v_j$ | $w_j$ |
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

---

##### Complete Model

$\displaystyle
\begin{align*}
\max\ & \sum_{i=1}^{15} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{15} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,15,\ j=1,\ldots,15
\end{align*}
$

Where the mapping of $j$ to ProductName, $v_j$, and $w_j$ is as listed above, and $i$ to PlatformID and $C_i$ is as listed above.