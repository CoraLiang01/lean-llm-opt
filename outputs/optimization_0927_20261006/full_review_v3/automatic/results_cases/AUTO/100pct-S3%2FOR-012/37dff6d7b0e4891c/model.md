Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let:
- $i$ index platforms, with resource_id from 1 to 10.
- $j$ index genres, with item_name as below.
- $v_j$ = item_value of genre $j$.
- $a_j$ = resource_requirement of genre $j$.
- $c_i$ = resource_capacity of platform $i$.

**Genres (item_name), Values ($v_j$), and Memory Requirements ($a_j$):**

| item_name    | item_value ($v_j$) | resource_requirement ($a_j$) |
|--------------|-------------------|------------------------------|
| Racing       | 28                | 393                          |
| Sports       | 69                | 195                          |
| Action       | 20                | 192                          |
| Adventure    | 62                | 155                          |
| RPG          | 58                | 500                          |
| Shooter      | 11                | 156                          |
| Strategy     | 73                | 317                          |
| Simulation   | 43                | 694                          |
| Puzzle       | 28                | 751                          |
| Fighting     | 57                | 467                          |
| Platformer   | 92                | 796                          |
| Survival     | 66                | 146                          |
| Horror       | 14                | 269                          |
| Sandbox      | 49                | 246                          |
| MMO          | 12                | 652                          |

**Platforms (resource_id) and Capacities ($c_i$):**

| resource_id | resource_capacity ($c_i$) |
|-------------|--------------------------|
| 1           | 1336                     |
| 2           | 1754                     |
| 3           | 1617                     |
| 4           | 1119                     |
| 5           | 1410                     |
| 6           | 627                      |
| 7           | 748                      |
| 8           | 1540                     |
| 9           | 1292                     |
| 10          | 1138                     |

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

For each platform $i$ (resource_id):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

Where:
- $v_j$ and $a_j$ are as listed above for each genre $j$.
- $c_i$ is as listed above for each platform $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All identifiers, coefficients, and constraints are as retrieved and required by the question.**