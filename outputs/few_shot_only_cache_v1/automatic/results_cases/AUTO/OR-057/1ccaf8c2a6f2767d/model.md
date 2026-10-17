Let $x_{ij}$ be the number of units of game $j$ (with genre/ProductName as in products.csv) to be listed on platform $i$ (PlatformID as in capacity.csv). All $x_{ij}$ are integer and $\geq 0$.

Let:
- $P$ = set of platforms (PlatformID from capacity.csv): $\{1,2,\ldots,15\}$
- $G$ = set of games/genres (ProductName from products.csv): $\{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$
- $v_j$ = Value of game $j$ (from products.csv)
- $w_j$ = Weight (memory requirement) of game $j$ (from products.csv)
- $C_i$ = Capacity of platform $i$ (from capacity.csv)

The model:

Maximize total value:
$$
\max \sum_{i \in P} \sum_{j \in G} v_j \cdot x_{ij}
$$

Subject to, for each platform $i \in P$:
$$
\sum_{j \in G} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in P
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in P,\, j \in G
$$

Where:

- $P = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $G = \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$
- $v_j$ and $w_j$ as:

| ProductName   | $v_j$ | $w_j$ |
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

- $C_i$ as:

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

All $x_{ij}$ are integer and $\geq 0$.