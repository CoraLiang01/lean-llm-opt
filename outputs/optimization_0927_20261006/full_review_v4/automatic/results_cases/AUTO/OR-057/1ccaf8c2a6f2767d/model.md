Let $x_{ij}$ be the number of units of game $j$ (ProductName) to be listed on platform $i$ (PlatformID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Platforms $i \in \{1,2,\ldots,15\}$, with capacities $C_i$:

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

- Games $j$ (ProductName), with value $v_j$ and memory requirement $w_j$:

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

**Mathematical Model**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\; j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{15} \sum_{j} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of game $j$.

**Constraints:**

For each platform $i$:

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,15
$$

where $w_j$ is the Weight (memory requirement) of game $j$, and $C_i$ is the Capacity of platform $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All identifiers and coefficients:**

- Platforms: PlatformID $i$ as listed above.
- Games: ProductName $j$ as listed above.
- Values $v_j$ and Weights $w_j$ as in the table.
- Platform capacities $C_i$ as in the table.

**Complete model:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{15} \sum_{j} v_j \cdot x_{ij} \\
\text{s.t.}\quad & \sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
$$

where $j$ runs over:

Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO

and $i$ runs over PlatformID $1$ to $15$ with the given capacities.