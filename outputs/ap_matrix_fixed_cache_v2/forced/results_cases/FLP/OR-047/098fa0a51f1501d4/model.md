##### Sets and Indices

- Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of platforms, indexed by $i$.
- Let $J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ be the set of game genres, indexed by $j$.

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

- Game genre values and memory requirements:
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

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of games from genre $j$ to be listed on platform $i$.

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

##### Constraints

1. **Platform memory capacity:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

##### Parameters (full listing)

- Platforms and capacities:
  - $I = \{1,2,3,4,5,6,7,8,9,10\}$
  - $C = [1336, 1754, 1617, 1119, 1410, 627, 748, 1540, 1292, 1138]$

- Genres, values, and weights:
  - $J = [\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}]$
  - $v = [28, 69, 20, 62, 58, 11, 73, 43, 28, 57, 92, 66, 14, 49, 12]$
  - $w = [393, 195, 192, 155, 500, 156, 317, 694, 751, 467, 796, 146, 269, 246, 652]$

##### Complete Mathematical Model

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \, x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{15} w_j \, x_{ij} \leq C_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\; j=1,\ldots,15
\end{align*}
\]

where $v_j$ and $w_j$ are as listed above for each genre $j$, and $C_i$ is the capacity for each platform $i$.