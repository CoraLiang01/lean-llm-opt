Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes platforms, with resource_id from capacity.csv.
- $j$ indexes genres, with item_name from products.csv.

**Parameters:**
- $v_j$: item_value of genre $j$ (from products.csv)
- $a_j$: resource_requirement of genre $j$ (from products.csv)
- $c_i$: resource_capacity of platform $i$ (from capacity.csv)

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

---

**Mathematical Model:**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
$$
where $v_j$ is the item_value for genre $j$ as above.

**Platform Capacity Constraints:**
$$
\sum_{j=1}^{15} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$
where $a_j$ is the resource_requirement for genre $j$ and $c_i$ is the resource_capacity for platform $i$ as above.

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**All identifiers and coefficients:**

- Platforms (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Platform capacities (resource_capacity): 1336, 1754, 1617, 1119, 1410, 627, 748, 1540, 1292, 1138
- Genres (item_name): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO
- Genre values (item_value): 28, 69, 20, 62, 58, 11, 73, 43, 28, 57, 92, 66, 14, 49, 12
- Genre memory requirements (resource_requirement): 393, 195, 192, 155, 500, 156, 317, 694, 751, 467, 796, 146, 269, 246, 652

---

**Complete Model:**

Maximize
$$
28x_{i,\text{Racing}} + 69x_{i,\text{Sports}} + 20x_{i,\text{Action}} + 62x_{i,\text{Adventure}} + 58x_{i,\text{RPG}} + 11x_{i,\text{Shooter}} + 73x_{i,\text{Strategy}} + 43x_{i,\text{Simulation}} + 28x_{i,\text{Puzzle}} + 57x_{i,\text{Fighting}} + 92x_{i,\text{Platformer}} + 66x_{i,\text{Survival}} + 14x_{i,\text{Horror}} + 49x_{i,\text{Sandbox}} + 12x_{i,\text{MMO}}
$$
summed over all $i = 1,\ldots,10$.

Subject to, for each platform $i$:
$$
393x_{i,\text{Racing}} + 195x_{i,\text{Sports}} + 192x_{i,\text{Action}} + 155x_{i,\text{Adventure}} + 500x_{i,\text{RPG}} + 156x_{i,\text{Shooter}} + 317x_{i,\text{Strategy}} + 694x_{i,\text{Simulation}} + 751x_{i,\text{Puzzle}} + 467x_{i,\text{Fighting}} + 796x_{i,\text{Platformer}} + 146x_{i,\text{Survival}} + 269x_{i,\text{Horror}} + 246x_{i,\text{Sandbox}} + 652x_{i,\text{MMO}} \leq c_i
$$
where $c_i$ is the resource_capacity for platform $i$ as listed above.

And
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$