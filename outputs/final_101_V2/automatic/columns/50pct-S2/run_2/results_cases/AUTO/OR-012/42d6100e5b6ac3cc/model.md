##### Sets and Indices

- Let $I$ be the set of platforms, indexed by $i$ (with resource_id from 1 to 10).
- Let $J$ be the set of genres, indexed by $j$ (with item_name as below).

##### Parameters

- $c_i$: memory capacity of platform $i$ (resource_capacity).
- $v_j$: value per unit of genre $j$ (item_value).
- $a_j$: memory requirement per unit of genre $j$ (resource_requirement).

##### Decision Variables

- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

##### Data

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

##### Mathematical Model

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

Subject to (for all $i \in I$):

Memory capacity on each platform:
$$
\sum_{j \in J} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
$$

Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

Where:

- $I = \{1,2,3,4,5,6,7,8,9,10\}$
- $J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$
- $c_i$ as given above for each $i$
- $v_j$ and $a_j$ as given above for each $j$