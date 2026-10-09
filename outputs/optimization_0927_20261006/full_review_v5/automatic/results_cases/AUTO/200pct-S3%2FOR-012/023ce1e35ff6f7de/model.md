#### Sets and Indices

- Let $I$ be the set of platforms, indexed by $i$, with platform identifiers given by the "resource_id" column in "capacity.csv".
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

Let $C_i$ denote the resource_capacity for platform $i$.

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

Let $v_j$ denote the item_value for genre $j$.

Let $a_j$ denote the resource_requirement (memory requirement) for genre $j$.

#### Decision Variables

- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$.
- Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

- **Platform Memory Capacity Constraints:**
  \[
  \sum_{j \in J} a_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
  \]

- **Nonnegativity and Integrality:**
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
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

**Summary:**  
Maximize the total value of games listed across all platforms, subject to each platform's memory capacity, by choosing nonnegative integer numbers of units of each genre to list on each platform. All parameters and identifiers are as retrieved and in source order.