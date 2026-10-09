Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

Let $I$ be the set of platforms, indexed by PlatformId:
$$
I = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}
$$

Let $J$ be the set of genres, indexed by ProductName:
$$
J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
$$

Let $v_j$ be the value of genre $j$ (from products.csv), and $w_j$ be the memory requirement of genre $j$.

Let $C_i$ be the memory capacity of platform $i$ (from capacity.csv).

**Objective:**
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i \in I$:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Parameter values:**

Platforms and capacities (from capacity.csv, in source order):

| PlatformId | Capacity |
|------------|----------|
| 1          | 1336     |
| 2          | 1754     |
| 3          | 1617     |
| 4          | 1119     |
| 5          | 1410     |
| 6          | 627      |
| 7          | 748      |
| 8          | 1540     |
| 9          | 1292     |
| 10         | 1138     |

Genres, values, and memory requirements (from products.csv, in source order):

| ProductName  | Value | Weight |
|--------------|-------|--------|
| Racing       | 28    | 393    |
| Sports       | 69    | 195    |
| Action       | 20    | 192    |
| Adventure    | 62    | 155    |
| RPG          | 58    | 500    |
| Shooter      | 11    | 156    |
| Strategy     | 73    | 317    |
| Simulation   | 43    | 694    |
| Puzzle       | 28    | 751    |
| Fighting     | 57    | 467    |
| Platformer   | 92    | 796    |
| Survival     | 66    | 146    |
| Horror       | 14    | 269    |
| Sandbox      | 49    | 246    |
| MMO          | 12    | 652    |

**Decision variables:**

For each platform $i$ and genre $j$, $x_{ij} \in \mathbb{Z}_{\geq 0}$.

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
\end{align*}
$$

Where the sets, parameters, and variable domains are as specified above.