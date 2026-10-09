Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Sets and Indices:**
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
\sum_{j} r_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
$$

where:
- $r_j$ is the resource_requirement (memory) for genre $j$
- $C_i$ is the resource_capacity for platform $i$

**Explicitly, for each platform:**

- For $i=1$: $\sum_{j} r_j x_{1j} \leq 1336$
- For $i=2$: $\sum_{j} r_j x_{2j} \leq 1754$
- For $i=3$: $\sum_{j} r_j x_{3j} \leq 1617$
- For $i=4$: $\sum_{j} r_j x_{4j} \leq 1119$
- For $i=5$: $\sum_{j} r_j x_{5j} \leq 1410$
- For $i=6$: $\sum_{j} r_j x_{6j} \leq 627$
- For $i=7$: $\sum_{j} r_j x_{7j} \leq 748$
- For $i=8$: $\sum_{j} r_j x_{8j} \leq 1540$
- For $i=9$: $\sum_{j} r_j x_{9j} \leq 1292$
- For $i=10$: $\sum_{j} r_j x_{10j} \leq 1138$

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All parameters and identifiers are as retrieved and preserved in original order.**