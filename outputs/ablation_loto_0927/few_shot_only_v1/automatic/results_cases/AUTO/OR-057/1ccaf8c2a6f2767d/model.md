**Mathematical Optimization Model**

Let:

- $i$ index the platforms, with PlatformID $i \in \{1,2,\ldots,15\}$.
- $j$ index the games/genres, with ProductName $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$.
- $x_{ij}$ = number of units of game $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$).

Parameters:

- $v_j$ = Value of one unit of game $j$ (from products.csv).
- $w_j$ = Memory requirement (Weight) of one unit of game $j$ (from products.csv).
- $C_i$ = Memory Capacity of platform $i$ (from capacity.csv).

---

**Objective:**

Maximize the total value of games listed across all platforms:
$$
\max \sum_{i=1}^{15} \sum_{j=1}^{15} v_j \cdot x_{ij}
$$

---

**Subject to:**

**1. Platform Memory Capacity Constraints:**

For each platform $i$:
$$
\sum_{j=1}^{15} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
$$

**2. Non-negativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,15\},\; j \in \{1,2,\ldots,15\}
$$

---

**Parameter Values (from source, in order):**

Platforms and Capacities (from capacity.csv):

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

Games/Genres, Values, and Weights (from products.csv):

| ProductName   | Value | Weight |
|---------------|-------|--------|
| Racing        | 59    | 776    |
| Sports        | 83    | 573    |
| Action        | 94    | 127    |
| Adventure     | 41    | 138    |
| RPG           | 96    | 385    |
| Shooter       | 12    | 263    |
| Strategy      | 83    | 473    |
| Simulation    | 36    | 387    |
| Puzzle        | 56    | 390    |
| Fighting      | 27    | 556    |
| Platformer    | 47    | 601    |
| Survival      | 24    | 441    |
| Horror        | 14    | 603    |
| Sandbox       | 22    | 411    |
| MMO           | 17    | 652    |

---

**Full Model (with explicit indices and coefficients):**

**Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \text{for } i = 1,\ldots,15;\; j = 1,\ldots,15
$$

**Objective:**
$$
\max \sum_{i=1}^{15} \Big[59x_{i1} + 83x_{i2} + 94x_{i3} + 41x_{i4} + 96x_{i5} + 12x_{i6} + 83x_{i7} + 36x_{i8} + 56x_{i9} + 27x_{i10} + 47x_{i11} + 24x_{i12} + 14x_{i13} + 22x_{i14} + 17x_{i15}\Big]
$$

**Subject to, for each $i = 1,\ldots,15$:**
$$
776x_{i1} + 573x_{i2} + 127x_{i3} + 138x_{i4} + 385x_{i5} + 263x_{i6} + 473x_{i7} + 387x_{i8} + 390x_{i9} + 556x_{i10} + 601x_{i11} + 441x_{i12} + 603x_{i13} + 411x_{i14} + 652x_{i15} \leq C_i
$$
where $C_i$ is the capacity for platform $i$ as listed above.

**And**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i,j
$$