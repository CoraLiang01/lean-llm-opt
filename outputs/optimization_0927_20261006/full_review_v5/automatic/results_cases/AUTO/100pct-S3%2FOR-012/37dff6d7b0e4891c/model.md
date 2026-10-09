#### Sets and Indices

- Let $I$ be the set of platforms, indexed by $i$, with platform IDs given by the "resource_id" column in "capacity.csv".
- Let $J$ be the set of genres, indexed by $j$, with genre names given by the "item_name" column in "products.csv".

#### Parameters

From "capacity.csv" (in source order):

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

From "products.csv" (in source order):

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

#### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of games from genre $j$ to be listed on platform $i$.

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]
where $v_j$ is the item_value for genre $j$.

**Subject to:**

- **Platform Capacity Constraints:** For each platform $i$,
\[
\sum_{j \in J} r_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
where $r_j$ is the resource_requirement for genre $j$, and $c_i$ is the resource_capacity for platform $i$.

- **Integrality and Nonnegativity:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Parameter Tables (source order)

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

**Summary of Model:**

- Maximize total value of listed games across all platforms.
- For each platform, total memory used by all listed games cannot exceed its resource_capacity.
- Decision variables are the number of units of each genre listed on each platform, and must be nonnegative integers.