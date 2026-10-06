Let $x_{ij}$ be the number of units of games from genre $j$ (item_name) to be listed on platform $i$ (resource_id). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Platforms $i$ (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Genres $j$ (item_name): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO
- $v_j$ (item_value): 
  - Racing: 28
  - Sports: 69
  - Action: 20
  - Adventure: 62
  - RPG: 58
  - Shooter: 11
  - Strategy: 73
  - Simulation: 43
  - Puzzle: 28
  - Fighting: 57
  - Platformer: 92
  - Survival: 66
  - Horror: 14
  - Sandbox: 49
  - MMO: 12
- $a_j$ (resource_requirement): 
  - Racing: 393
  - Sports: 195
  - Action: 192
  - Adventure: 155
  - RPG: 500
  - Shooter: 156
  - Strategy: 317
  - Simulation: 694
  - Puzzle: 751
  - Fighting: 467
  - Platformer: 796
  - Survival: 146
  - Horror: 269
  - Sandbox: 246
  - MMO: 652
- $c_i$ (resource_capacity): 
  - 1: 1336
  - 2: 1754
  - 3: 1617
  - 4: 1119
  - 5: 1410
  - 6: 627
  - 7: 748
  - 8: 1540
  - 9: 1292
  - 10: 1138

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

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Where:**

- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$ (integer, $\geq 0$)
- $v_j$: Value of genre $j$ (see above)
- $a_j$: Memory requirement of genre $j$ (see above)
- $c_i$: Memory capacity of platform $i$ (see above)

All identifiers and coefficients are as retrieved and preserved in source order.