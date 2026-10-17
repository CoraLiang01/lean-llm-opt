Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

**Indices:**
- $i$ indexes platforms, with PlatformId as below.
- $j$ indexes genres, with ProductName as below.

**Parameters:**

- $v_j$: Value of one unit of genre $j$ game (from Value column)
- $w_j$: Memory requirement of one unit of genre $j$ game (from Weight column)
- $C_i$: Memory capacity of platform $i$ (from Capacity column)

**Data:**

Platforms (from capacity.csv):

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

Genres (from products.csv):

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

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{PlatformId}\},\ j \in \{\text{ProductName}\}
$$

**Objective:**

$$
\max \sum_{i \in \{\text{PlatformId}\}} \sum_{j \in \{\text{ProductName}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i$ (PlatformId):

$$
\sum_{j \in \{\text{ProductName}\}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{PlatformId}\}
$$

Where:

- $v_j$ and $w_j$ are as in the table above for each genre $j$.
- $C_i$ is as in the table above for each platform $i$.

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All identifiers and coefficients are as retrieved.**