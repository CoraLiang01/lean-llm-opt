Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Platforms (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Platform capacities (resource_capacity):

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

- Genres (item_name) and their value/memory requirement:

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

**Decision Variables:**

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for each platform $i$ (resource_id) and genre $j$ (item_name).

---

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the item_value for genre $j$.

---

**Constraints:**

For each platform $i$ (resource_id):

$$
\sum_{j} r_j \cdot x_{ij} \leq c_i
$$

where $r_j$ is the resource_requirement for genre $j$, and $c_i$ is the resource_capacity for platform $i$.

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Numerical Formulation:**

Let $i$ index resource_id (platforms), $j$ index item_name (genres).

- $c_i$ (resource_capacity):

  $c_1 = 1336$, $c_2 = 1754$, $c_3 = 1617$, $c_4 = 1119$, $c_5 = 1410$, $c_6 = 627$, $c_7 = 748$, $c_8 = 1540$, $c_9 = 1292$, $c_{10} = 1138$

- $v_j$ (item_value), $r_j$ (resource_requirement):

  - Racing: $v = 28$, $r = 393$
  - Sports: $v = 69$, $r = 195$
  - Action: $v = 20$, $r = 192$
  - Adventure: $v = 62$, $r = 155$
  - RPG: $v = 58$, $r = 500$
  - Shooter: $v = 11$, $r = 156$
  - Strategy: $v = 73$, $r = 317$
  - Simulation: $v = 43$, $r = 694$
  - Puzzle: $v = 28$, $r = 751$
  - Fighting: $v = 57$, $r = 467$
  - Platformer: $v = 92$, $r = 796$
  - Survival: $v = 66$, $r = 146$
  - Horror: $v = 14$, $r = 269$
  - Sandbox: $v = 49$, $r = 246$
  - MMO: $v = 12$, $r = 652$

**Full Model:**

$$
\max \sum_{i=1}^{10} \Bigg[
28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}}
\Bigg]
$$

Subject to, for each $i = 1,\ldots,10$:

$$
393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq c_i
$$

and

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
$$

where $c_i$ is the resource_capacity for platform $i$ as listed above.