Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Parameters:**

- Platforms (from capacity.csv, in source order):

  | resource_id |
  |-------------|
  | 1           |
  | 2           |
  | 3           |
  | 4           |
  | 5           |
  | 6           |
  | 7           |
  | 8           |
  | 9           |
  | 10          |

  Each platform $i$ has memory capacity $C_i$:

  $$
  \begin{align*}
  C_1 &= 1336 \\
  C_2 &= 1754 \\
  C_3 &= 1617 \\
  C_4 &= 1119 \\
  C_5 &= 1410 \\
  C_6 &= 627 \\
  C_7 &= 748 \\
  C_8 &= 1540 \\
  C_9 &= 1292 \\
  C_{10} &= 1138 \\
  \end{align*}
  $$

- Genres (from products.csv, in source order):

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

Let $G$ be the set of genres (in the above order), and $P$ the set of platforms (resource_id as above).

Let $v_j$ be the value of genre $j$, and $a_j$ its memory requirement.

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in P, \forall j \in G
$$

**Objective:**

$$
\max \sum_{i \in P} \sum_{j \in G} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i \in P$:

$$
\sum_{j \in G} a_j \cdot x_{ij} \leq C_i
$$

For all $i \in P$, $j \in G$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Parameter Table (source order):**

- Platforms and capacities:

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

- Genres, values, and memory requirements:

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

- Maximize total value of games listed across all platforms.
- For each platform, total memory used by all genres cannot exceed its capacity.
- Decision variables are the number of units of each genre listed on each platform, and must be nonnegative integers.