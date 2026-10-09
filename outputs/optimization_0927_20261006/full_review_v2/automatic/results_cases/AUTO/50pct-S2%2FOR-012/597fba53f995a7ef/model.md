Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (platforms, identified by resource_id)
- $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (genres, identified by item_name)

**Parameters:**

From capacity.csv:
- Platform 1: resource_capacity = 1336
- Platform 2: resource_capacity = 1754
- Platform 3: resource_capacity = 1617
- Platform 4: resource_capacity = 1119
- Platform 5: resource_capacity = 1410
- Platform 6: resource_capacity = 627
- Platform 7: resource_capacity = 748
- Platform 8: resource_capacity = 1540
- Platform 9: resource_capacity = 1292
- Platform 10: resource_capacity = 1138

From products.csv (for each genre $j$):
- Racing: item_value = 28, resource_requirement = 393
- Sports: item_value = 69, resource_requirement = 195
- Action: item_value = 20, resource_requirement = 192
- Adventure: item_value = 62, resource_requirement = 155
- RPG: item_value = 58, resource_requirement = 500
- Shooter: item_value = 11, resource_requirement = 156
- Strategy: item_value = 73, resource_requirement = 317
- Simulation: item_value = 43, resource_requirement = 694
- Puzzle: item_value = 28, resource_requirement = 751
- Fighting: item_value = 57, resource_requirement = 467
- Platformer: item_value = 92, resource_requirement = 796
- Survival: item_value = 66, resource_requirement = 146
- Horror: item_value = 14, resource_requirement = 269
- Sandbox: item_value = 49, resource_requirement = 246
- MMO: item_value = 12, resource_requirement = 652

---

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j} v_j \cdot x_{ij}
$$
where $v_j$ is the item_value for genre $j$.

**Constraints:**

For each platform $i$ (resource_id):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

where:
- $a_j$ is the resource_requirement (memory) for genre $j$
- $c_i$ is the resource_capacity for platform $i$

**Explicitly, for each platform:**

- Platform 1:
  $$
  393x_{1,\text{Racing}} + 195x_{1,\text{Sports}} + 192x_{1,\text{Action}} + 155x_{1,\text{Adventure}} + 500x_{1,\text{RPG}} + 156x_{1,\text{Shooter}} + 317x_{1,\text{Strategy}} + 694x_{1,\text{Simulation}} + 751x_{1,\text{Puzzle}} + 467x_{1,\text{Fighting}} + 796x_{1,\text{Platformer}} + 146x_{1,\text{Survival}} + 269x_{1,\text{Horror}} + 246x_{1,\text{Sandbox}} + 652x_{1,\text{MMO}} \leq 1336
  $$
- Platform 2:
  $$
  393x_{2,\text{Racing}} + 195x_{2,\text{Sports}} + 192x_{2,\text{Action}} + 155x_{2,\text{Adventure}} + 500x_{2,\text{RPG}} + 156x_{2,\text{Shooter}} + 317x_{2,\text{Strategy}} + 694x_{2,\text{Simulation}} + 751x_{2,\text{Puzzle}} + 467x_{2,\text{Fighting}} + 796x_{2,\text{Platformer}} + 146x_{2,\text{Survival}} + 269x_{2,\text{Horror}} + 246x_{2,\text{Sandbox}} + 652x_{2,\text{MMO}} \leq 1754
  $$
- Platform 3:
  $$
  393x_{3,\text{Racing}} + 195x_{3,\text{Sports}} + 192x_{3,\text{Action}} + 155x_{3,\text{Adventure}} + 500x_{3,\text{RPG}} + 156x_{3,\text{Shooter}} + 317x_{3,\text{Strategy}} + 694x_{3,\text{Simulation}} + 751x_{3,\text{Puzzle}} + 467x_{3,\text{Fighting}} + 796x_{3,\text{Platformer}} + 146x_{3,\text{Survival}} + 269x_{3,\text{Horror}} + 246x_{3,\text{Sandbox}} + 652x_{3,\text{MMO}} \leq 1617
  $$
- Platform 4:
  $$
  393x_{4,\text{Racing}} + 195x_{4,\text{Sports}} + 192x_{4,\text{Action}} + 155x_{4,\text{Adventure}} + 500x_{4,\text{RPG}} + 156x_{4,\text{Shooter}} + 317x_{4,\text{Strategy}} + 694x_{4,\text{Simulation}} + 751x_{4,\text{Puzzle}} + 467x_{4,\text{Fighting}} + 796x_{4,\text{Platformer}} + 146x_{4,\text{Survival}} + 269x_{4,\text{Horror}} + 246x_{4,\text{Sandbox}} + 652x_{4,\text{MMO}} \leq 1119
  $$
- Platform 5:
  $$
  393x_{5,\text{Racing}} + 195x_{5,\text{Sports}} + 192x_{5,\text{Action}} + 155x_{5,\text{Adventure}} + 500x_{5,\text{RPG}} + 156x_{5,\text{Shooter}} + 317x_{5,\text{Strategy}} + 694x_{5,\text{Simulation}} + 751x_{5,\text{Puzzle}} + 467x_{5,\text{Fighting}} + 796x_{5,\text{Platformer}} + 146x_{5,\text{Survival}} + 269x_{5,\text{Horror}} + 246x_{5,\text{Sandbox}} + 652x_{5,\text{MMO}} \leq 1410
  $$
- Platform 6:
  $$
  393x_{6,\text{Racing}} + 195x_{6,\text{Sports}} + 192x_{6,\text{Action}} + 155x_{6,\text{Adventure}} + 500x_{6,\text{RPG}} + 156x_{6,\text{Shooter}} + 317x_{6,\text{Strategy}} + 694x_{6,\text{Simulation}} + 751x_{6,\text{Puzzle}} + 467x_{6,\text{Fighting}} + 796x_{6,\text{Platformer}} + 146x_{6,\text{Survival}} + 269x_{6,\text{Horror}} + 246x_{6,\text{Sandbox}} + 652x_{6,\text{MMO}} \leq 627
  $$
- Platform 7:
  $$
  393x_{7,\text{Racing}} + 195x_{7,\text{Sports}} + 192x_{7,\text{Action}} + 155x_{7,\text{Adventure}} + 500x_{7,\text{RPG}} + 156x_{7,\text{Shooter}} + 317x_{7,\text{Strategy}} + 694x_{7,\text{Simulation}} + 751x_{7,\text{Puzzle}} + 467x_{7,\text{Fighting}} + 796x_{7,\text{Platformer}} + 146x_{7,\text{Survival}} + 269x_{7,\text{Horror}} + 246x_{7,\text{Sandbox}} + 652x_{7,\text{MMO}} \leq 748
  $$
- Platform 8:
  $$
  393x_{8,\text{Racing}} + 195x_{8,\text{Sports}} + 192x_{8,\text{Action}} + 155x_{8,\text{Adventure}} + 500x_{8,\text{RPG}} + 156x_{8,\text{Shooter}} + 317x_{8,\text{Strategy}} + 694x_{8,\text{Simulation}} + 751x_{8,\text{Puzzle}} + 467x_{8,\text{Fighting}} + 796x_{8,\text{Platformer}} + 146x_{8,\text{Survival}} + 269x_{8,\text{Horror}} + 246x_{8,\text{Sandbox}} + 652x_{8,\text{MMO}} \leq 1540
  $$
- Platform 9:
  $$
  393x_{9,\text{Racing}} + 195x_{9,\text{Sports}} + 192x_{9,\text{Action}} + 155x_{9,\text{Adventure}} + 500x_{9,\text{RPG}} + 156x_{9,\text{Shooter}} + 317x_{9,\text{Strategy}} + 694x_{9,\text{Simulation}} + 751x_{9,\text{Puzzle}} + 467x_{9,\text{Fighting}} + 796x_{9,\text{Platformer}} + 146x_{9,\text{Survival}} + 269x_{9,\text{Horror}} + 246x_{9,\text{Sandbox}} + 652x_{9,\text{MMO}} \leq 1292
  $$
- Platform 10:
  $$
  393x_{10,\text{Racing}} + 195x_{10,\text{Sports}} + 192x_{10,\text{Action}} + 155x_{10,\text{Adventure}} + 500x_{10,\text{RPG}} + 156x_{10,\text{Shooter}} + 317x_{10,\text{Strategy}} + 694x_{10,\text{Simulation}} + 751x_{10,\text{Puzzle}} + 467x_{10,\text{Fighting}} + 796x_{10,\text{Platformer}} + 146x_{10,\text{Survival}} + 269x_{10,\text{Horror}} + 246x_{10,\text{Sandbox}} + 652x_{10,\text{MMO}} \leq 1138
  $$

**Variable domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All coefficients, identifiers, and constraints are as retrieved and in original order.**