Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let:
- $i$ index platforms, with resource_id from capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$ index genres, with item_name from products.csv: $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$

Parameters:
- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$
- $c_i$ = resource_capacity of platform $i$

Numerical values:

Platforms (from capacity.csv, in order):
- resource_id: 1, resource_capacity: 1336
- resource_id: 2, resource_capacity: 1754
- resource_id: 3, resource_capacity: 1617
- resource_id: 4, resource_capacity: 1119
- resource_id: 5, resource_capacity: 1410
- resource_id: 6, resource_capacity: 627
- resource_id: 7, resource_capacity: 748
- resource_id: 8, resource_capacity: 1540
- resource_id: 9, resource_capacity: 1292
- resource_id: 10, resource_capacity: 1138

Genres (from products.csv, in order):

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

The model:

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
$$

Subject to, for each platform $i$ (resource_id as below):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
$$

That is, explicitly:

- For resource_id 1: $\sum_{j} a_j x_{1j} \leq 1336$
- For resource_id 2: $\sum_{j} a_j x_{2j} \leq 1754$
- For resource_id 3: $\sum_{j} a_j x_{3j} \leq 1617$
- For resource_id 4: $\sum_{j} a_j x_{4j} \leq 1119$
- For resource_id 5: $\sum_{j} a_j x_{5j} \leq 1410$
- For resource_id 6: $\sum_{j} a_j x_{6j} \leq 627$
- For resource_id 7: $\sum_{j} a_j x_{7j} \leq 748$
- For resource_id 8: $\sum_{j} a_j x_{8j} \leq 1540$
- For resource_id 9: $\sum_{j} a_j x_{9j} \leq 1292$
- For resource_id 10: $\sum_{j} a_j x_{10j} \leq 1138$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Where:
- $v_j$ and $a_j$ are as given above for each genre $j$.
- $c_i$ is as given above for each platform $i$.

All variables, parameters, and constraints use the original identifiers and coefficients as retrieved.