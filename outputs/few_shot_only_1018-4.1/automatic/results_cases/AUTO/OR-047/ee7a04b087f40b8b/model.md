Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$, where $i$ indexes PlatformId and $j$ indexes ProductName (genre). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Platforms ($i$):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10

- Genres ($j$) and their values and memory requirements:

| ProductName | Value | Weight |
|-------------|-------|--------|
| Racing      | 28    | 393    |
| Sports      | 69    | 195    |
| Action      | 20    | 192    |
| Adventure   | 62    | 155    |
| RPG         | 58    | 500    |
| Shooter     | 11    | 156    |
| Strategy    | 73    | 317    |
| Simulation  | 43    | 694    |
| Puzzle      | 28    | 751    |
| Fighting    | 57    | 467    |
| Platformer  | 92    | 796    |
| Survival    | 66    | 146    |
| Horror      | 14    | 269    |
| Sandbox     | 49    | 246    |
| MMO         | 12    | 652    |

- Platform capacities:

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

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j \in \text{Genres}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of genre $j$.

**Subject to:**

For each platform $i$:
\[
\sum_{j \in \text{Genres}} w_j \cdot x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10
\]
where $w_j$ is the Weight (memory requirement) of genre $j$, and $c_i$ is the Capacity of platform $i$.

**Variable domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\; j \in \text{Genres}
\]

**Where:**

- Genres = {Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}
- $v_j$ and $w_j$ as given above for each genre $j$
- $c_i$ as given above for each platform $i$