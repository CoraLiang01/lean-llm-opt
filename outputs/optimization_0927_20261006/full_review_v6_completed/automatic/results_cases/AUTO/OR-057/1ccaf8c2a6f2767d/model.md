Let $x_{ij}$ be the number of units of game $j$ (genre) to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

**Sets and Indices:**

- $i \in \{\text{1}, \text{2}, \ldots, \text{15}\}$ (PlatformID from capacity.csv)
- $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ (ProductName from products.csv)

**Parameters:**

- $c_i$ = Capacity of platform $i$ (from capacity.csv)
- $v_j$ = Value of game $j$ (from products.csv)
- $w_j$ = Weight (memory requirement) of game $j$ (from products.csv)

**Data:**

Platform capacities (in source order):

| PlatformID | Capacity |
|------------|----------|
| 1          | 995      |
| 2          | 1143     |
| 3          | 949      |
| 4          | 969      |
| 5          | 1649     |
| 6          | 870      |
| 7          | 1064     |
| 8          | 536      |
| 9          | 766      |
| 10         | 532      |
| 11         | 1703     |
| 12         | 1633     |
| 13         | 1203     |
| 14         | 1979     |
| 15         | 1797     |

Game values and memory requirements (in source order):

| ProductName  | Value | Weight |
|--------------|-------|--------|
| Racing       | 59    | 776    |
| Sports       | 83    | 573    |
| Action       | 94    | 127    |
| Adventure    | 41    | 138    |
| RPG          | 96    | 385    |
| Shooter      | 12    | 263    |
| Strategy     | 83    | 473    |
| Simulation   | 36    | 387    |
| Puzzle       | 56    | 390    |
| Fighting     | 27    | 556    |
| Platformer   | 47    | 601    |
| Survival     | 24    | 441    |
| Horror       | 14    | 603    |
| Sandbox      | 22    | 411    |
| MMO          | 17    | 652    |

---

### Mathematical Model

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Racing}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i$:

$$
\sum_{j} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,15\}
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Where:**

- $v_j$ and $w_j$ are as given in the table above for each genre $j$.
- $c_i$ is as given in the table above for each platform $i$.

---

**All identifiers, coefficients, and constraints are preserved as in the source data.**