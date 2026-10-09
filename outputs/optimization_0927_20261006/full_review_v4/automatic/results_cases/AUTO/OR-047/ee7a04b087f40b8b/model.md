Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Sets and Indices:**

- $i$ indexes platforms, with PlatformId from capacity.csv.
- $j$ indexes game genres, with ProductName from products.csv.

**Parameters:**

- $c_i$: Capacity of platform $i$ (from capacity.csv).
- $v_j$: Value per unit of genre $j$ (from products.csv).
- $w_j$: Memory requirement per unit of genre $j$ (from products.csv).

**Data:**

Platforms (from capacity.csv, in file order):

| PlatformId |
|------------|
| 1          |
| 2          |
| 3          |
| 4          |
| 5          |
| 6          |
| 7          |
| 8          |
| 9          |
| 10         |

Capacities:

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

Genres and their parameters (from products.csv, in file order):

| ProductName   | $v_j$ | $w_j$ |
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

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j \cdot x_{ij}
\]

Subject to, for each platform $i$:

\[
\sum_{j} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Where:**

- $v_j$ and $w_j$ are as listed above for each genre $j$.
- $c_i$ is as listed above for each platform $i$.
- $x_{ij}$ is the integer number of units of genre $j$ to list on platform $i$.

**All data and indices are preserved in original file and row order.**