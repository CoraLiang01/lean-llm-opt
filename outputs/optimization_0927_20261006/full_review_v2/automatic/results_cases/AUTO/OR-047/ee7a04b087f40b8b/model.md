Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let:
- $i$ index platforms, with PlatformId from the capacity.csv file.
- $j$ index genres, with ProductName from the products.csv file.
- $v_j$ = Value of genre $j$ (from products.csv).
- $w_j$ = Weight (memory requirement) of genre $j$ (from products.csv).
- $C_i$ = Capacity of platform $i$ (from capacity.csv).

#### Sets and Parameters

Platforms ($i$):
- 1: $C_1 = 1336$
- 2: $C_2 = 1754$
- 3: $C_3 = 1617$
- 4: $C_4 = 1119$
- 5: $C_5 = 1410$
- 6: $C_6 = 627$
- 7: $C_7 = 748$
- 8: $C_8 = 1540$
- 9: $C_9 = 1292$
- 10: $C_{10} = 1138$

Genres ($j$) and their parameters:
- Racing: $v_{\text{Racing}} = 28$, $w_{\text{Racing}} = 393$
- Sports: $v_{\text{Sports}} = 69$, $w_{\text{Sports}} = 195$
- Action: $v_{\text{Action}} = 20$, $w_{\text{Action}} = 192$
- Adventure: $v_{\text{Adventure}} = 62$, $w_{\text{Adventure}} = 155$
- RPG: $v_{\text{RPG}} = 58$, $w_{\text{RPG}} = 500$
- Shooter: $v_{\text{Shooter}} = 11$, $w_{\text{Shooter}} = 156$
- Strategy: $v_{\text{Strategy}} = 73$, $w_{\text{Strategy}} = 317$
- Simulation: $v_{\text{Simulation}} = 43$, $w_{\text{Simulation}} = 694$
- Puzzle: $v_{\text{Puzzle}} = 28$, $w_{\text{Puzzle}} = 751$
- Fighting: $v_{\text{Fighting}} = 57$, $w_{\text{Fighting}} = 467$
- Platformer: $v_{\text{Platformer}} = 92$, $w_{\text{Platformer}} = 796$
- Survival: $v_{\text{Survival}} = 66$, $w_{\text{Survival}} = 146$
- Horror: $v_{\text{Horror}} = 14$, $w_{\text{Horror}} = 269$
- Sandbox: $v_{\text{Sandbox}} = 49$, $w_{\text{Sandbox}} = 246$
- MMO: $v_{\text{MMO}} = 12$, $w_{\text{MMO}} = 652$

#### Mathematical Model

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j \cdot x_{ij}
$$

Subject to, for each platform $i$:
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

#### Data Used

Platforms and capacities (from capacity.csv, in file order):

| PlatformId | Capacity |
|------------|----------|
| 1          | 1336     |
| 2          | 1754     |
| 3          | 1617     |
| 4          | 1119     |
| 5          | 1410     |
| 6          | 627      |
| 7          | 748      |
| 8          | 1540     |
| 9          | 1292     |
| 10         | 1138     |

Genres and parameters (from products.csv, in file order):

| ProductName  | Value | Weight |
|--------------|-------|--------|
| Racing       | 28    | 393    |
| Sports       | 69    | 195    |
| Action       | 20    | 192    |
| Adventure    | 62    | 155    |
| RPG          | 58    | 500    |
| Shooter      | 11    | 156    |
| Strategy     | 73    | 317    |
| Simulation   | 43    | 694    |
| Puzzle       | 28    | 751    |
| Fighting     | 57    | 467    |
| Platformer   | 92    | 796    |
| Survival     | 66    | 146    |
| Horror       | 14    | 269    |
| Sandbox      | 49    | 246    |
| MMO          | 12    | 652    |