#### Sets and Indices

- Let $I$ be the set of platforms, indexed by $i$, with platform IDs given by the "resource_id" column in "capacity.csv".
- Let $J$ be the set of genres, indexed by $j$, with genre names given by the "item_name" column in "products.csv".

#### Parameters

From "capacity.csv":

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

From "products.csv":

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

Let $c_i$ be the resource_capacity for platform $i$.

Let $v_j$ be the item_value for genre $j$.

Let $a_j$ be the resource_requirement for genre $j$.

#### Decision Variables

- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$.
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

#### Objective Function

Maximize the total value of games listed across all platforms:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

1. **Platform Memory Capacity Constraints** (for each platform $i$):
   $$
   \sum_{j \in J} a_j \cdot x_{ij} \leq c_i, \quad \forall i \in I
   $$

2. **Nonnegativity and Integrality**:
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   $$

#### Parameter Tables

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

#### Complete Model

Maximize:
$$
\sum_{i=1}^{10} \sum_{j \in J} v_j \cdot x_{ij}
$$

Subject to, for each $i = 1, \ldots, 10$:
$$
\sum_{j \in J} a_j \cdot x_{ij} \leq c_i
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i, j
$$

where $c_i$, $v_j$, and $a_j$ are as given in the tables above.