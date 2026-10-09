**Mathematical Optimization Model**

Let:

- $i$ index the platforms, with PlatformId $i \in \{1,2,3,4,5,6,7,8,9,10\}$.
- $j$ index the game genres, with genre $j$ corresponding to ProductName in the order given.
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$).

Parameters:

- $v_j$ = value of one unit of genre $j$ (from Value column).
- $w_j$ = memory requirement (Weight) of one unit of genre $j$.
- $C_i$ = memory capacity of platform $i$.

**Data:**

Platforms and capacities:

| PlatformId | $C_i$ |
|:----------:|:-----:|
| 1          | 1336  |
| 2          | 1754  |
| 3          | 1617  |
| 4          | 1119  |
| 5          | 1410  |
| 6          | 627   |
| 7          | 748   |
| 8          | 1540  |
| 9          | 1292  |
| 10         | 1138  |

Genres, values, and weights:

| $j$ | ProductName   | $v_j$ | $w_j$ |
|:---:|:-------------|:-----:|:-----:|
| 1   | Racing       | 28    | 393   |
| 2   | Sports       | 69    | 195   |
| 3   | Action       | 20    | 192   |
| 4   | Adventure    | 62    | 155   |
| 5   | RPG          | 58    | 500   |
| 6   | Shooter      | 11    | 156   |
| 7   | Strategy     | 73    | 317   |
| 8   | Simulation   | 43    | 694   |
| 9   | Puzzle       | 28    | 751   |
| 10  | Fighting     | 57    | 467   |
| 11  | Platformer   | 92    | 796   |
| 12  | Survival     | 66    | 146   |
| 13  | Horror       | 14    | 269   |
| 14  | Sandbox      | 49    | 246   |
| 15  | MMO          | 12    | 652   |

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i$:

$$
\sum_{j=1}^{15} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,15\}
$$

---

**Where:**

- $v_j$ and $w_j$ are as listed above for each genre $j$.
- $C_i$ is as listed above for each platform $i$.
- $x_{ij}$ is the integer number of units of genre $j$ to list on platform $i$.