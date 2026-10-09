Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Sets and Indices:**

- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (platforms, identified by resource_id)
- $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ (genres, identified by item_name)

**Parameters:**

- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$
- $c_i$ = resource_capacity of platform $i$

**Data:**

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

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
\]

**Subject to:**

- **Platform Capacity Constraints:**
  \[
  \sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
  \]

- **Nonnegativity and Integrality:**
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
  \]

**Where:**

- $v_j$ and $a_j$ are as given in the table above for each genre $j$.
- $c_i$ is as given in the table above for each platform $i$.

**Decision variables:**

- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$ (integer, $\geq 0$).