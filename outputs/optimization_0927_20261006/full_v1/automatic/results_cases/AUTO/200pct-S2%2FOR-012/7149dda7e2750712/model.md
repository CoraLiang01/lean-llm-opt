Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let $i$ index platforms, with platform IDs given by the "resource_id" column in capacity.csv:

Platforms ($i$): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

Let $j$ index genres, with genre names given by the "item_name" column in products.csv:

Genres ($j$): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO

Let $v_j$ be the value of genre $j$ ("item_value" in products.csv).

Let $a_j$ be the memory requirement of genre $j$ ("resource_requirement" in products.csv).

Let $C_i$ be the memory capacity of platform $i$ ("resource_capacity" in capacity.csv).

The complete model is:

---

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$:
\[
\sum_{j} a_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Parameter Data (in source order):**

Platforms (from capacity.csv):

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

Genres (from products.csv):

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

**Full Model:**

\[
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij}
\]

Subject to, for each $i=1,\ldots,10$:
\[
393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq C_i
\]

where $C_i$ is as above for each platform.

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
\]