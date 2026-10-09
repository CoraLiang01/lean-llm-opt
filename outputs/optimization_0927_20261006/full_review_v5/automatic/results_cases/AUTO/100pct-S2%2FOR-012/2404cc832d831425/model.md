Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes platforms, with resource_id from capacity.csv.
- $j$ indexes game genres, with item_name from products.csv.

**Parameters:**

From capacity.csv:

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

From products.csv:

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

Let $v_j$ be the item_value for genre $j$, and $a_j$ be the resource_requirement for genre $j$. Let $c_i$ be the resource_capacity for platform $i$.

---

### Mathematical Model

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i$ (resource_id):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Parameter Tables (as retrieved):**

Platforms (capacity.csv):

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

Genres (products.csv):

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

**Decision variables:**

$x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$, integer and $\geq 0$.

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{15} a_j x_{ij} \leq c_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10,\; j=1,\ldots,15
\end{align*}
$$

Where $v_j$ and $a_j$ are as in the tables above, and $c_i$ as per platform.