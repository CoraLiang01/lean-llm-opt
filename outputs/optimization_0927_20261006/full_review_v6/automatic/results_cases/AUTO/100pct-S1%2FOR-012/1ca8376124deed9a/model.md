Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let:
- $i$ index platforms, with platform IDs as in the "resource_id" column of capacity.csv.
- $j$ index genres, with genre names as in the "item_name" column of products.csv.
- $v_j$ = item_value of genre $j$ (from products.csv)
- $a_j$ = resource_requirement of genre $j$ (from products.csv)
- $c_i$ = resource_capacity of platform $i$ (from capacity.csv)

**Objective:**
\[
\max \sum_{i \in \{\text{1,2,3,4,5,6,7,8,9,10}\}} \sum_{j \in \{\text{Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$ (resource_id):

\[
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{\text{1,2,3,4,5,6,7,8,9,10}\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Parameter values (in source order):**

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

**Decision variables:**
\[
x_{ij} = \text{number of units of genre } j \text{ to be listed on platform } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t. } & \sum_{j=1}^{15} a_j x_{ij} \leq c_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\ j=1,\ldots,15
\end{align*}
\]

Where the mapping of $i$ and $j$ to platform and genre is as above, and $v_j$, $a_j$, $c_i$ are as listed.