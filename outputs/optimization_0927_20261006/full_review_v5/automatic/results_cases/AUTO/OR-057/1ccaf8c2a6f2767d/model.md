Let $x_{ij}$ be the number of units of game $j$ (genre) to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Platforms $i$ (PlatformID):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

- Platform capacities $C_i$:
  - $C_1 = 995$
  - $C_2 = 1143$
  - $C_3 = 949$
  - $C_4 = 969$
  - $C_5 = 1649$
  - $C_6 = 870$
  - $C_7 = 1064$
  - $C_8 = 536$
  - $C_9 = 766$
  - $C_{10} = 532$
  - $C_{11} = 1703$
  - $C_{12} = 1633$
  - $C_{13} = 1203$
  - $C_{14} = 1979$
  - $C_{15} = 1797$

- Games $j$ (ProductName), with value $v_j$ and memory requirement $w_j$:

| ProductName   | $v_j$ | $w_j$ |
|---------------|-------|-------|
| Racing        | 59    | 776   |
| Sports        | 83    | 573   |
| Action        | 94    | 127   |
| Adventure     | 41    | 138   |
| RPG           | 96    | 385   |
| Shooter       | 12    | 263   |
| Strategy      | 83    | 473   |
| Simulation    | 36    | 387   |
| Puzzle        | 56    | 390   |
| Fighting      | 27    | 556   |
| Platformer    | 47    | 601   |
| Survival      | 24    | 441   |
| Horror        | 14    | 603   |
| Sandbox       | 22    | 411   |
| MMO           | 17    | 652   |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{15} \sum_{j} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i$:
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Parameter Tables (as retrieved):**

Platforms and Capacities:

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

Games, Values, and Memory Requirements:

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