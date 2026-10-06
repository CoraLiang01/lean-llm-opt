##### Sets and Indices

- Let $I$ be the set of platforms, indexed by $i$ (resource_id from capacity.csv).
- Let $J$ be the set of genres, indexed by $j$ (item_name from products.csv).

##### Parameters

From capacity.csv:
- $c_i$ = resource_capacity of platform $i$

From products.csv:
- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$

##### Decision Variables

- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

##### Objective Function

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

##### Constraints

1. **Platform Memory Capacity Constraints** (for each platform $i$):

$$
\sum_{j \in J} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
$$

2. **Non-negativity and Integrality**:

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

#### Data

**Platforms (from capacity.csv):**

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

**Genres (from products.csv):**

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

##### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j \in J} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in J} a_j \cdot x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\; j \in J
\end{align*}
$$

Where:

- $c_i$ is the resource_capacity for platform $i$ (see table above)
- $v_j$ is the item_value for genre $j$ (see table above)
- $a_j$ is the resource_requirement for genre $j$ (see table above)
- $x_{ij}$ is the integer number of units of genre $j$ to list on platform $i$