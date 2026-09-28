##### Sets

- $I = \{1,2,3,4,5,6,7,8,9,10\}$: set of platforms (PlatformId from capacity.csv)
- $J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$: set of game genres (ProductName from products.csv)

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
  - $\text{Racing}$: $v_{\text{Racing}} = 28$, $w_{\text{Racing}} = 393$
  - $\text{Sports}$: $v_{\text{Sports}} = 69$, $w_{\text{Sports}} = 195$
  - $\text{Action}$: $v_{\text{Action}} = 20$, $w_{\text{Action}} = 192$
  - $\text{Adventure}$: $v_{\text{Adventure}} = 62$, $w_{\text{Adventure}} = 155$
  - $\text{RPG}$: $v_{\text{RPG}} = 58$, $w_{\text{RPG}} = 500$
  - $\text{Shooter}$: $v_{\text{Shooter}} = 11$, $w_{\text{Shooter}} = 156$
  - $\text{Strategy}$: $v_{\text{Strategy}} = 73$, $w_{\text{Strategy}} = 317$
  - $\text{Simulation}$: $v_{\text{Simulation}} = 43$, $w_{\text{Simulation}} = 694$
  - $\text{Puzzle}$: $v_{\text{Puzzle}} = 28$, $w_{\text{Puzzle}} = 751$
  - $\text{Fighting}$: $v_{\text{Fighting}} = 57$, $w_{\text{Fighting}} = 467$
  - $\text{Platformer}$: $v_{\text{Platformer}} = 92$, $w_{\text{Platformer}} = 796$
  - $\text{Survival}$: $v_{\text{Survival}} = 66$, $w_{\text{Survival}} = 146$
  - $\text{Horror}$: $v_{\text{Horror}} = 14$, $w_{\text{Horror}} = 269$
  - $\text{Sandbox}$: $v_{\text{Sandbox}} = 49$, $w_{\text{Sandbox}} = 246$
  - $\text{MMO}$: $v_{\text{MMO}} = 12$, $w_{\text{MMO}} = 652$

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of games from genre $j \in J$ to be listed on platform $i \in I$

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. Platform memory capacity:
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]
2. Integer and nonnegativity:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

##### Complete Parameter Listing

- Platforms and capacities:
  - $I = \{1,2,3,4,5,6,7,8,9,10\}$
  - $C = [1336, 1754, 1617, 1119, 1410, 627, 748, 1540, 1292, 1138]$
- Genres, values, and weights:
  - $J = [\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}]$
  - $v = [28, 69, 20, 62, 58, 11, 73, 43, 28, 57, 92, 66, 14, 49, 12]$
  - $w = [393, 195, 192, 155, 500, 156, 317, 694, 751, 467, 796, 146, 269, 246, 652]$