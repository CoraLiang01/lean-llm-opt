**Sets:**
- Let $I$ be the set of platforms, indexed by $i$ (PlatformID from "capacity.csv").
- Let $J$ be the set of games/genres, indexed by $j$ (ProductName from "products.csv").

**Parameters:**
- $c_i$: Capacity of platform $i$ (from "capacity.csv", column "Capacity").
- $v_j$: Value of game $j$ (from "products.csv", column "Value").
- $w_j$: Memory requirement of game $j$ (from "products.csv", column "Weight").

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of game $j$ to be listed on platform $i$.

**Model:**

Maximize total value:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

Subject to, for each platform $i \in I$:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i
$$

And
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

**Numerical Data:**

Platforms ($i$) and their capacities ($c_i$):

| PlatformID | Capacity ($c_i$) |
|------------|------------------|
| 1          | 995              |
| 2          | 1143             |
| 3          | 949              |
| 4          | 969              |
| 5          | 1649             |
| 6          | 870              |
| 7          | 1064             |
| 8          | 536              |
| 9          | 766              |
| 10         | 532              |
| 11         | 1703             |
| 12         | 1633             |
| 13         | 1203             |
| 14         | 1979             |
| 15         | 1797             |

Games/Genres ($j$) and their values ($v_j$) and memory requirements ($w_j$):

| ProductName   | Value ($v_j$) | Weight ($w_j$) |
|---------------|--------------|---------------|
| Racing        | 59           | 776           |
| Sports        | 83           | 573           |
| Action        | 94           | 127           |
| Adventure     | 41           | 138           |
| RPG           | 96           | 385           |
| Shooter       | 12           | 263           |
| Strategy      | 83           | 473           |
| Simulation    | 36           | 387           |
| Puzzle        | 56           | 390           |
| Fighting      | 27           | 556           |
| Platformer    | 47           | 601           |
| Survival      | 24           | 441           |
| Horror        | 14           | 603           |
| Sandbox       | 22           | 411           |
| MMO           | 17           | 652           |

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{15} \sum_{j=1}^{15} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{15} w_j \cdot x_{ij} \leq c_i, \quad \forall i = 1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,15,\; j = 1,\ldots,15
\end{align*}
$$

Where:

- $v_j$ and $w_j$ are as listed above for each ProductName.
- $c_i$ is as listed above for each PlatformID.