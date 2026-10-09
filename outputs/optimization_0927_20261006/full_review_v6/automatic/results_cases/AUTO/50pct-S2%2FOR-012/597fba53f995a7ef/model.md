Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let:
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ index the platforms, with resource IDs as given in capacity.csv.
- $j$ index the genres, with names as given in products.csv.

Define:
- $c_i$ = resource_capacity of platform $i$
- $v_j$ = item_value of genre $j$
- $a_j$ = resource_requirement of genre $j$

The genres and their parameters are:

| Genre         | $v_j$ | $a_j$ |
|---------------|-------|-------|
| Racing        | 28    | 393   |
| Sports        | 69    | 195   |
| Action        | 20    | 192   |
| Adventure     | 62    | 155   |
| RPG           | 58    | 500   |
| Shooter       | 11    | 156   |
| Strategy      | 73    | 317   |
| Simulation    | 43    | 694   |
| Puzzle        | 28    | 751   |
| Fighting      | 57    | 467   |
| Platformer    | 92    | 796   |
| Survival      | 66    | 146   |
| Horror        | 14    | 269   |
| Sandbox       | 49    | 246   |
| MMO           | 12    | 652   |

The platforms and their capacities are:

| Platform $i$ | resource_id | $c_i$ |
|--------------|-------------|-------|
| 1            | 1           | 1336  |
| 2            | 2           | 1754  |
| 3            | 3           | 1617  |
| 4            | 4           | 1119  |
| 5            | 5           | 1410  |
| 6            | 6           | 627   |
| 7            | 7           | 748   |
| 8            | 8           | 1540  |
| 9            | 9           | 1292  |
| 10           | 10          | 1138  |

The mathematical model is:

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j \in \text{Genres}} v_j \cdot x_{ij}
$$

Subject to, for each platform $i$:
$$
\sum_{j \in \text{Genres}} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\; \forall j \in \text{Genres}
$$

Where:
- $v_j$, $a_j$ as above for each genre $j$
- $c_i$ as above for each platform $i$
- $x_{ij}$ is the integer number of units of games from genre $j$ to be listed on platform $i$