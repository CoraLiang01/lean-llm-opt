Let $x_{ij}$ be the number of units of genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

Let $i$ index platforms (resource_id from capacity.csv), $j$ index genres (item_name from products.csv).

Let $v_j$ be the value per unit of genre $j$ (item_value), $a_j$ be the memory requirement per unit of genre $j$ (resource_requirement), and $C_i$ be the memory capacity of platform $i$ (resource_capacity).

**Sets:**

- Platforms $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- Genres $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$

**Parameters:**

From capacity.csv (in source order):

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

From products.csv (in source order):

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
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
$$

Subject to, for each platform $i$:

$$
\sum_{j=1}^{15} a_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,15\}
$$

**Where:**

- $v_j$ and $a_j$ are as given in the products.csv table above, in source order.
- $C_i$ is as given in the capacity.csv table above, in source order.
- $x_{ij}$ is the number of units of genre $j$ to be listed on platform $i$.

**All data and indices are preserved in original file and row order.**