Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes platforms: $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (PlatformId from capacity.csv)
- $j$ indexes genres: $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ (ProductName from products.csv)

**Parameters:**
- $v_j$ = Value of genre $j$ (from products.csv)
- $w_j$ = Weight (memory requirement) of genre $j$ (from products.csv)
- $C_i$ = Capacity of platform $i$ (from capacity.csv)

**Data:**

Platforms and capacities (in source order):

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

Genres, values, and weights (in source order):

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
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j} v_j \cdot x_{ij}
$$

where $v_j$ is the value for genre $j$ as given above.

**Constraints:**

For each platform $i$:

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the weight (memory requirement) for genre $j$ and $C_i$ is the capacity for platform $i$ as given above.

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All identifiers, coefficients, and constraints are as retrieved and in original order.**