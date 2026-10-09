Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

Let $i$ index platforms (resource_id from capacity.csv), and $j$ index genres (item_name from products.csv).

Let $v_j$ be the value (item_value) of genre $j$.
Let $a_j$ be the memory requirement (resource_requirement) of genre $j$.
Let $C_i$ be the memory capacity (resource_capacity) of platform $i$.

**Objective:**
\[
\max \sum_{i \in \{1,2,3,4,5,6,7,8,9,10\}} \sum_{j \in \text{Genres}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$:
\[
\sum_{j \in \text{Genres}} a_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Data:**

Platforms (resource_id, resource_capacity):

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

Genres (item_name, item_value, resource_requirement):

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

**Variables:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}\}
\]

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{15} a_j x_{ij} \leq C_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,15
\end{align*}
\]

Where $v_j$ and $a_j$ are as listed above for each genre $j$, and $C_i$ is as listed above for each platform $i$.