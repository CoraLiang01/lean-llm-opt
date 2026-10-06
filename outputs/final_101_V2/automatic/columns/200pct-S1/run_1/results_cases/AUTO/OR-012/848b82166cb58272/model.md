Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Sets and Indices:**
- Platforms $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (from resource_id in capacity.csv)
- Genres $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (from item_name in products.csv)

**Parameters:**
- $c_i$: Memory capacity of platform $i$ (resource_capacity from capacity.csv)
- $v_j$: Value per unit of genre $j$ (item_value from products.csv)
- $a_j$: Memory requirement per unit of genre $j$ (resource_requirement from products.csv)

**Data:**

Platform capacities ($c_i$):

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

Game genres, values, and memory requirements:

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

---

**Mathematical Model**

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \text{Genres}} v_j \cdot x_{ij}
$$

Subject to, for each platform $i$:
$$
\sum_{j \in \text{Genres}} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Where:
- $v_j$ and $a_j$ are as given in the table above for each genre $j$.
- $c_i$ is as given above for each platform $i$.

All data and constraints are included as required.