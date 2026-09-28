##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of game genre $j$ to be listed on platform $i$.

Where:
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (platforms)
- $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (game genres)

##### Parameters

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

- Game genre values and weights:
  - Racing: $v_{\text{Racing}} = 28$, $w_{\text{Racing}} = 393$
  - Sports: $v_{\text{Sports}} = 69$, $w_{\text{Sports}} = 195$
  - Action: $v_{\text{Action}} = 20$, $w_{\text{Action}} = 192$
  - Adventure: $v_{\text{Adventure}} = 62$, $w_{\text{Adventure}} = 155$
  - RPG: $v_{\text{RPG}} = 58$, $w_{\text{RPG}} = 500$
  - Shooter: $v_{\text{Shooter}} = 11$, $w_{\text{Shooter}} = 156$
  - Strategy: $v_{\text{Strategy}} = 73$, $w_{\text{Strategy}} = 317$
  - Simulation: $v_{\text{Simulation}} = 43$, $w_{\text{Simulation}} = 694$
  - Puzzle: $v_{\text{Puzzle}} = 28$, $w_{\text{Puzzle}} = 751$
  - Fighting: $v_{\text{Fighting}} = 57$, $w_{\text{Fighting}} = 467$
  - Platformer: $v_{\text{Platformer}} = 92$, $w_{\text{Platformer}} = 796$
  - Survival: $v_{\text{Survival}} = 66$, $w_{\text{Survival}} = 146$
  - Horror: $v_{\text{Horror}} = 14$, $w_{\text{Horror}} = 269$
  - Sandbox: $v_{\text{Sandbox}} = 49$, $w_{\text{Sandbox}} = 246$
  - MMO: $v_{\text{MMO}} = 12$, $w_{\text{MMO}} = 652$

##### Objective Function

$\max \sum_{i=1}^{10} \sum_{j \in G} v_j x_{ij}$

where $G$ is the set of all genres listed above.

##### Constraints

For each platform $i = 1,\ldots,10$:
$$
\sum_{j \in G} w_j x_{ij} \leq C_i
$$

For all $i$ and $j$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

##### Complete Model

Let $G = \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$.

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \big[28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}} \big] \\
\text{s.t.}\quad
& 393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq C_i, \\
& \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10,\ j \in G
\end{align*}
$$

Where the capacities $C_i$ are:
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