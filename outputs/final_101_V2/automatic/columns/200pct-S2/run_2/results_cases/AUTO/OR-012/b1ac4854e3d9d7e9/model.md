Let $x_{ij}$ be the number of units of games from genre $j$ (item_name) to be listed on platform $i$ (resource_id). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Platforms $i$ (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Genres $j$ (item_name): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO
- Value of genre $j$: $v_j$ (item_value)
- Memory requirement of genre $j$: $a_j$ (resource_requirement)
- Capacity of platform $i$: $c_i$ (resource_capacity)

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all genres}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$ (resource_id):

\[
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Parameter Values (from data):**

Platforms and capacities:
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

Genres, values, and memory requirements:
- Racing: value 28, requirement 393
- Sports: value 69, requirement 195
- Action: value 20, requirement 192
- Adventure: value 62, requirement 155
- RPG: value 58, requirement 500
- Shooter: value 11, requirement 156
- Strategy: value 73, requirement 317
- Simulation: value 43, requirement 694
- Puzzle: value 28, requirement 751
- Fighting: value 57, requirement 467
- Platformer: value 92, requirement 796
- Survival: value 66, requirement 146
- Horror: value 14, requirement 269
- Sandbox: value 49, requirement 246
- MMO: value 12, requirement 652

---

**Complete Model:**

\[
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij}
\]

Subject to, for each $i$:

\[
393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq c_i
\]

where $c_i$ is the resource_capacity for platform $i$ as listed above.

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]