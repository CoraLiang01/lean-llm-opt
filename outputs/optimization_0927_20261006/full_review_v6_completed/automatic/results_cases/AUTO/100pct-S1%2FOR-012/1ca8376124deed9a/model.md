Let $x_{ij}$ be the number of units of genre $j$ to be listed on platform $i$. All $x_{ij}$ are nonnegative integers.

**Sets and Indices:**
- Platforms $i \in \{1,2,3,4,5,6,7,8,9,10\}$, identified by resource_id.
- Genres $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$, identified by item_name.

**Parameters:**
- $v_j$: value per unit of genre $j$ (item_value).
- $a_j$: memory requirement per unit of genre $j$ (resource_requirement).
- $C_i$: memory capacity of platform $i$ (resource_capacity).

**Data:**

Platforms (from capacity.csv, in source order):

| resource_id | resource_capacity |
|-------------|------------------|
| 1           | 1336             |
| 2           | 1754             |
| 3           | 1617             |
| 4           | 1119             |
| 5           | 1410             |
| 6           | 627              |
| 7           | 748              |
| 8           | 1540             |
| 9           | 1292             |
| 10          | 1138             |

Genres (from products.csv, in source order):

| item_name   | item_value | resource_requirement |
|-------------|------------|---------------------|
| Racing      | 28         | 393                 |
| Sports      | 69         | 195                 |
| Action      | 20         | 192                 |
| Adventure   | 62         | 155                 |
| RPG         | 58         | 500                 |
| Shooter     | 11         | 156                 |
| Strategy    | 73         | 317                 |
| Simulation  | 43         | 694                 |
| Puzzle      | 28         | 751                 |
| Fighting    | 57         | 467                 |
| Platformer  | 92         | 796                 |
| Survival    | 66         | 146                 |
| Horror      | 14         | 269                 |
| Sandbox     | 49         | 246                 |
| MMO         | 12         | 652                 |

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
\]

Subject to, for each platform $i$ (resource_id):

\[
\sum_{j} a_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Where:**

- $v_j$ and $a_j$ are as given in the table above for each genre $j$.
- $C_i$ is as given in the table above for each platform $i$.

**All variables and constraints use the original identifiers and coefficients as retrieved.**