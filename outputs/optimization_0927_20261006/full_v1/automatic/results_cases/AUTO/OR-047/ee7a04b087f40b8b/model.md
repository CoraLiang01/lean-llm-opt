Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let $P$ be the set of platforms (indexed by PlatformId), and $G$ be the set of genres (indexed by ProductName).

Let $v_j$ be the value of genre $j$, and $w_j$ be the memory requirement (Weight) of genre $j$.

Let $C_i$ be the memory capacity of platform $i$.

---

**Sets:**

- Platforms $P = \{1,2,3,4,5,6,7,8,9,10\}$
- Genres $G = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$

**Parameters:**

- Platform capacities:
  - $C_1 = 1336$
  - $C_2 = 1754$
  - $C_3 = 1617$
  - $C_4 = 1119$
  - $C_5 = 1410$
  - $C_6 = 627$
  - $C_7 = 748$
  - $C_8 = 1540$
  - $C_9 = 1292$
  - $C_{10} = 1138$

- Genre values and memory requirements:

| Genre        | $v_j$ (Value) | $w_j$ (Weight) |
|--------------|:-------------:|:--------------:|
| Racing       | 28            | 393            |
| Sports       | 69            | 195            |
| Action       | 20            | 192            |
| Adventure    | 62            | 155            |
| RPG          | 58            | 500            |
| Shooter      | 11            | 156            |
| Strategy     | 73            | 317            |
| Simulation   | 43            | 694            |
| Puzzle       | 28            | 751            |
| Fighting     | 57            | 467            |
| Platformer   | 92            | 796            |
| Survival     | 66            | 146            |
| Horror       | 14            | 269            |
| Sandbox      | 49            | 246            |
| MMO          | 12            | 652            |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in P} \sum_{j \in G} v_j \cdot x_{ij}
\]

**Subject to:**

- Platform memory capacity constraints:
  \[
  \sum_{j \in G} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in P
  \]

- Non-negativity and integrality:
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in P,\, j \in G
  \]

---

**All data used:**

- Platforms (PlatformId): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Platform capacities: 1336, 1754, 1617, 1119, 1410, 627, 748, 1540, 1292, 1138
- Genres (ProductName): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO
- Genre values: 28, 69, 20, 62, 58, 11, 73, 43, 28, 57, 92, 66, 14, 49, 12
- Genre memory requirements: 393, 195, 192, 155, 500, 156, 317, 694, 751, 467, 796, 146, 269, 246, 652

**Decision variables:**

- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$, integer and nonnegative.