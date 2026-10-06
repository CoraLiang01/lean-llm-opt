Let $x_{ij}$ be the number of units of games from genre $j$ (item_name) to be listed on platform $i$ (resource_id). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Platforms $i$ (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Genres $j$ (item_name): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO
- $v_j$: item_value for genre $j$
- $a_j$: resource_requirement for genre $j$
- $c_i$: resource_capacity for platform $i$

**Data:**

Platforms and capacities:
- 1: $c_1 = 1336$
- 2: $c_2 = 1754$
- 3: $c_3 = 1617$
- 4: $c_4 = 1119$
- 5: $c_5 = 1410$
- 6: $c_6 = 627$
- 7: $c_7 = 748$
- 8: $c_8 = 1540$
- 9: $c_9 = 1292$
- 10: $c_{10} = 1138$

Genres, values, and requirements:
- Racing: $v_{\text{Racing}} = 28$, $a_{\text{Racing}} = 393$
- Sports: $v_{\text{Sports}} = 69$, $a_{\text{Sports}} = 195$
- Action: $v_{\text{Action}} = 20$, $a_{\text{Action}} = 192$
- Adventure: $v_{\text{Adventure}} = 62$, $a_{\text{Adventure}} = 155$
- RPG: $v_{\text{RPG}} = 58$, $a_{\text{RPG}} = 500$
- Shooter: $v_{\text{Shooter}} = 11$, $a_{\text{Shooter}} = 156$
- Strategy: $v_{\text{Strategy}} = 73$, $a_{\text{Strategy}} = 317$
- Simulation: $v_{\text{Simulation}} = 43$, $a_{\text{Simulation}} = 694$
- Puzzle: $v_{\text{Puzzle}} = 28$, $a_{\text{Puzzle}} = 751$
- Fighting: $v_{\text{Fighting}} = 57$, $a_{\text{Fighting}} = 467$
- Platformer: $v_{\text{Platformer}} = 92$, $a_{\text{Platformer}} = 796$
- Survival: $v_{\text{Survival}} = 66$, $a_{\text{Survival}} = 146$
- Horror: $v_{\text{Horror}} = 14$, $a_{\text{Horror}} = 269$
- Sandbox: $v_{\text{Sandbox}} = 49$, $a_{\text{Sandbox}} = 246$
- MMO: $v_{\text{MMO}} = 12$, $a_{\text{MMO}} = 652$

---

**Mathematical Model**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j \in \text{Genres}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$:
\[
\sum_{j \in \text{Genres}} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; j \in \text{Genres}
\]

**Where:**

- $v_j$ and $a_j$ are as listed above for each genre $j$.
- $c_i$ is as listed above for each platform $i$.
- $x_{ij}$ is the integer number of units of games from genre $j$ to be listed on platform $i$.