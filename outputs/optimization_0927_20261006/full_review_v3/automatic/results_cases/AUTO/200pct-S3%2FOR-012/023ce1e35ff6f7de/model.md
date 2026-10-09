Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Sets and Indices:**
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (platforms, identified by resource_id)
- $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ (genres, identified by item_name)

**Parameters:**
- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$
- $c_i$ = resource_capacity of platform $i$

**Data:**

Platforms (resource_id, resource_capacity):
- 1: 1336
- 2: 1754
- 3: 1617
- 4: 1119
- 5: 1410
- 6: 627
- 7: 748
- 8: 1540
- 9: 1292
- 10: 1138

Genres (item_name, item_value, resource_requirement):
- Racing: 28, 393
- Sports: 69, 195
- Action: 20, 192
- Adventure: 62, 155
- RPG: 58, 500
- Shooter: 11, 156
- Strategy: 73, 317
- Simulation: 43, 694
- Puzzle: 28, 751
- Fighting: 57, 467
- Platformer: 92, 796
- Survival: 66, 146
- Horror: 14, 269
- Sandbox: 49, 246
- MMO: 12, 652

---

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i$:
$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

Where:
- $v_j$ and $a_j$ are as listed above for each genre $j$
- $c_i$ is as listed above for each platform $i$

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**Retrieved Data Used:**

Platforms:
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

Genres:
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