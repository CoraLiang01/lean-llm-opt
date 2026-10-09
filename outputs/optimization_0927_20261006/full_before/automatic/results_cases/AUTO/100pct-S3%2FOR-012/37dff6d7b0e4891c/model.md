Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

Let:
- $i$ index platforms, with resource_id from 1 to 10.
- $j$ index genres, with item_name as below.
- $v_j$ = item_value of genre $j$.
- $a_j$ = resource_requirement of genre $j$.
- $c_i$ = resource_capacity of platform $i$.

#### Sets and Parameters

Platforms (resource_id) and their capacities:
- 1: $c_1 = 1336$
- 2: $c_2 = 1754$
- 3: $c_3 = 1617$
- 4: $c_4 = 1119$
- 5: $c_5 = 1410$
- 6: $c_6 = 627$
- 7: $c_7 = 748$
- 8: $c_8 = 1540$
- 9: $c_9 = 1292$
- 10: $c_{10} = 1138$

Genres (item_name), values ($v_j$), and memory requirements ($a_j$):

| Genre        | $v_j$ | $a_j$ |
|--------------|-------|-------|
| Racing       | 28    | 393   |
| Sports       | 69    | 195   |
| Action       | 20    | 192   |
| Adventure    | 62    | 155   |
| RPG          | 58    | 500   |
| Shooter      | 11    | 156   |
| Strategy     | 73    | 317   |
| Simulation   | 43    | 694   |
| Puzzle       | 28    | 751   |
| Fighting     | 57    | 467   |
| Platformer   | 92    | 796   |
| Survival     | 66    | 146   |
| Horror       | 14    | 269   |
| Sandbox      | 49    | 246   |
| MMO          | 12    | 652   |

#### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all platforms $i$ (resource_id 1–10) and genres $j$ (as above).

#### Objective

$$
\max \sum_{i=1}^{10} \sum_{j \in \text{Genres}} v_j \cdot x_{ij}
$$

#### Constraints

For each platform $i$ (resource_id 1–10):

$$
\sum_{j \in \text{Genres}} a_j \cdot x_{ij} \leq c_i
\qquad \forall i \in \{1,2,\ldots,10\}
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

#### Data Used

Platforms (resource_id) and resource_capacity:
- 1: 1336
- 2: 1754
- 3: 1617
- 4: 1119
- 5: 1410
- 6: 627
- 7: 748
- 8: 1540
- 9: 1292
- 10: 1138

Genres (item_name), item_value, resource_requirement:
- Racing: 28, 393
- Sports: 69, 195
- Action: 20, 192
- Adventure: 62, 155
- RPG: 58, 500
- Shooter: 11, 156
- Strategy: 73, 317
- Simulation: 43, 694
- Puzzle: 28, 751
- Fighting: 57, 467
- Platformer: 92, 796
- Survival: 66, 146
- Horror: 14, 269
- Sandbox: 49, 246
- MMO: 12, 652

#### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j \in \text{Genres}} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in \text{Genres}} a_j \cdot x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
$$

where $v_j$ and $a_j$ are as listed above, $c_i$ is the resource_capacity for platform $i$, and $x_{ij}$ is the integer number of units of genre $j$ to list on platform $i$.