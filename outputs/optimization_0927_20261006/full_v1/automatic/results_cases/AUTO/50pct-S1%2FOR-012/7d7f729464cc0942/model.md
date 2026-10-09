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

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\; j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
$$

where $v_j$ is the item_value for genre $j$ as listed above.

**Constraints:**

For each platform $i$ (resource_id as below):

$$
\sum_{j=1}^{15} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $a_j$ is the resource_requirement for genre $j$ and $c_i$ is the resource_capacity for platform $i$ as listed above.

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**Explicit Data Mapping:**

- Platforms $i$ (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Genres $j$ (item_name): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO
- $v_j$ (item_value): 28, 69, 20, 62, 58, 11, 73, 43, 28, 57, 92, 66, 14, 49, 12
- $a_j$ (resource_requirement): 393, 195, 192, 155, 500, 156, 317, 694, 751, 467, 796, 146, 269, 246, 652
- $c_i$ (resource_capacity): 1336, 1754, 1617, 1119, 1410, 627, 748, 1540, 1292, 1138

---

**Complete Model:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{15} a_j x_{ij} \leq c_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\; j=1,\ldots,15
\end{align*}
$$

with all coefficients and identifiers as above.