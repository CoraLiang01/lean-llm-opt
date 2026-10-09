#### Sets and Indices

- Let $I$ be the set of platforms, indexed by $i$, with platform IDs given by the column "resource_id" in "capacity.csv".
- Let $J$ be the set of genres, indexed by $j$, with genre names given by the column "item_name" in "products.csv".

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

Define:
- $c_i$ = resource_capacity of platform $i$
- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$

#### Decision Variables

- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$  
  ($x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in I$, $j \in J$)

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

- **Platform Capacity Constraints:**  
  For each platform $i \in I$,
  \[
  \sum_{j \in J} a_j \cdot x_{ij} \leq c_i
  \]

- **Nonnegativity and Integrality:**
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
  \]

---

#### Parameter Tables

**Platforms (from capacity.csv, source order):**

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

**Genres (from products.csv, source order):**

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
Maximize the total value of games listed across all platforms, subject to each platform's memory capacity, by choosing integer numbers of units of each genre to list on each platform.