Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes platforms, with resource_id from 1 to 10.
- $j$ indexes genres, with item_name as below.

**Parameters:**
- $v_j$: item_value of genre $j$
- $a_j$: resource_requirement (memory) of genre $j$
- $c_i$: resource_capacity of platform $i$

**Data:**

Platforms (resource_id, resource_capacity):

| $i$ | resource_id | resource_capacity |
|-----|-------------|------------------|
| 1   | 1           | 1336             |
| 2   | 2           | 1754             |
| 3   | 3           | 1617             |
| 4   | 4           | 1119             |
| 5   | 5           | 1410             |
| 6   | 6           | 627              |
| 7   | 7           | 748              |
| 8   | 8           | 1540             |
| 9   | 9           | 1292             |
| 10  | 10          | 1138             |

Genres (item_name, item_value, resource_requirement):

| $j$           | item_name    | item_value | resource_requirement |
|---------------|-------------|------------|----------------------|
| 1             | Racing      | 28         | 393                  |
| 2             | Sports      | 69         | 195                  |
| 3             | Action      | 20         | 192                  |
| 4             | Adventure   | 62         | 155                  |
| 5             | RPG         | 58         | 500                  |
| 6             | Shooter     | 11         | 156                  |
| 7             | Strategy    | 73         | 317                  |
| 8             | Simulation  | 43         | 694                  |
| 9             | Puzzle      | 28         | 751                  |
| 10            | Fighting    | 57         | 467                  |
| 11            | Platformer  | 92         | 796                  |
| 12            | Survival    | 66         | 146                  |
| 13            | Horror      | 14         | 269                  |
| 14            | Sandbox     | 49         | 246                  |
| 15            | MMO         | 12         | 652                  |

---

**Mathematical Model:**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,15\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i$ (resource_id):

$$
\sum_{j=1}^{15} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

Where:
- $v_j$ and $a_j$ are as in the table above for each genre $j$,
- $c_i$ is as in the table above for each platform $i$.

**Variable domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

**All coefficients and identifiers are as retrieved above.**