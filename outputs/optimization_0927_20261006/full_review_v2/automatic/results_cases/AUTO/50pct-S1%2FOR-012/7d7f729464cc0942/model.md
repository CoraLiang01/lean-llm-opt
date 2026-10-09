Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes platforms, with resource_id from capacity.csv.
- $j$ indexes genres, with item_name from products.csv.

**Parameters:**
- $v_j$: value per unit of genre $j$ (item_value from products.csv)
- $a_j$: memory requirement per unit of genre $j$ (resource_requirement from products.csv)
- $c_i$: memory capacity of platform $i$ (resource_capacity from capacity.csv)

**Data:**

Platforms (from capacity.csv, in source order):

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

Genres (from products.csv, in source order):

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

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij}
$$
where $v_j$ is the item_value for genre $j$ as above.

**Constraints:**

For each platform $i$ (resource_id as below):

$$
\sum_{j=1}^{15} a_j x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
$$

where $a_j$ is the resource_requirement for genre $j$, and $c_i$ is the resource_capacity for platform $i$.

**Explicitly:**

For each $i$:

- $c_1 = 1336$
- $c_2 = 1754$
- $c_3 = 1617$
- $c_4 = 1119$
- $c_5 = 1410$
- $c_6 = 627$
- $c_7 = 748$
- $c_8 = 1540$
- $c_9 = 1292$
- $c_{10} = 1138$

For each $j$ (in source order):

| $j$ | Genre        | $v_j$ | $a_j$ |
|-----|-------------|-------|-------|
| 1   | Racing      | 28    | 393   |
| 2   | Sports      | 69    | 195   |
| 3   | Action      | 20    | 192   |
| 4   | Adventure   | 62    | 155   |
| 5   | RPG         | 58    | 500   |
| 6   | Shooter     | 11    | 156   |
| 7   | Strategy    | 73    | 317   |
| 8   | Simulation  | 43    | 694   |
| 9   | Puzzle      | 28    | 751   |
| 10  | Fighting    | 57    | 467   |
| 11  | Platformer  | 92    | 796   |
| 12  | Survival    | 66    | 146   |
| 13  | Horror      | 14    | 269   |
| 14  | Sandbox     | 49    | 246   |
| 15  | MMO         | 12    | 652   |

**Summary:**

Maximize
$$
\sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij}
$$

Subject to, for each $i=1,\ldots,10$:
$$
\sum_{j=1}^{15} a_j x_{ij} \leq c_i
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i,j
$$

where all parameters are as listed above, and all indices and coefficients are preserved in source order.