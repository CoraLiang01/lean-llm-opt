##### Sets and Indices

Let:
- $I$ = set of platforms, indexed by $i$ (from capacity.csv, resource_id)
- $J$ = set of genres, indexed by $j$ (from products.csv, item_name)

##### Parameters

From capacity.csv:
- $C_i$ = resource_capacity of platform $i$

From products.csv:
- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$

##### Decision Variables

- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

##### Mathematical Model

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

Subject to:

Platform memory capacity constraints:
$$
\sum_{j \in J} a_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
$$

Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

##### Data

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

##### Complete Model

$$
\begin{align*}
\max\ & \sum_{i \in \{1,\ldots,10\}} \sum_{j \in J} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j \in J} a_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\, j \in J
\end{align*}
$$

Where:
- $C_i$ is as listed above for each resource_id,
- $v_j$ and $a_j$ are as listed above for each item_name.