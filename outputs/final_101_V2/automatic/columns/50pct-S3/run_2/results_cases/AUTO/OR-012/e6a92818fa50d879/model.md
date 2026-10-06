Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Parameters:**

- Platforms $i$ (resource_id from capacity.csv):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10

- Genres $j$ (item_name from products.csv):  
  Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO

- Platform capacities $c_i$ (resource_capacity from capacity.csv):  
  $c_1 = 1336$, $c_2 = 1754$, $c_3 = 1617$, $c_4 = 1119$, $c_5 = 1410$, $c_6 = 627$, $c_7 = 748$, $c_8 = 1540$, $c_9 = 1292$, $c_{10} = 1138$

- Game values $v_j$ (item_value from products.csv):  
  Racing: 28, Sports: 69, Action: 20, Adventure: 62, RPG: 58, Shooter: 11, Strategy: 73, Simulation: 43, Puzzle: 28, Fighting: 57, Platformer: 92, Survival: 66, Horror: 14, Sandbox: 49, MMO: 12

- Game memory requirements $a_j$ (resource_requirement from products.csv):  
  Racing: 393, Sports: 195, Action: 192, Adventure: 155, RPG: 500, Shooter: 156, Strategy: 317, Simulation: 694, Puzzle: 751, Fighting: 467, Platformer: 796, Survival: 146, Horror: 269, Sandbox: 246, MMO: 652

---

**Mathematical Model**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$:
\[
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Parameter Table (for reference):**

| Platform (resource_id) | Capacity ($c_i$) |
|------------------------|------------------|
| 1                      | 1336             |
| 2                      | 1754             |
| 3                      | 1617             |
| 4                      | 1119             |
| 5                      | 1410             |
| 6                      | 627              |
| 7                      | 748              |
| 8                      | 1540             |
| 9                      | 1292             |
| 10                     | 1138             |

| Genre (item_name) | Value ($v_j$) | Memory ($a_j$) |
|-------------------|--------------|---------------|
| Racing            | 28           | 393           |
| Sports            | 69           | 195           |
| Action            | 20           | 192           |
| Adventure         | 62           | 155           |
| RPG               | 58           | 500           |
| Shooter           | 11           | 156           |
| Strategy          | 73           | 317           |
| Simulation        | 43           | 694           |
| Puzzle            | 28           | 751           |
| Fighting          | 57           | 467           |
| Platformer        | 92           | 796           |
| Survival          | 66           | 146           |
| Horror            | 14           | 269           |
| Sandbox           | 49           | 246           |
| MMO               | 12           | 652           |