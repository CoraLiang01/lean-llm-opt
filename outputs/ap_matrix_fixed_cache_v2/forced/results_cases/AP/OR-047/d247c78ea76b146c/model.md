##### Sets

- $I$ = set of platforms = $\{1,2,3,4,5,6,7,8,9,10\}$
- $J$ = set of genres = $\{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$

##### Parameters

- $C_i$ = capacity (memory limit) of platform $i$:

  $C_1 = 1336$, $C_2 = 1754$, $C_3 = 1617$, $C_4 = 1119$, $C_5 = 1410$, $C_6 = 627$, $C_7 = 748$, $C_8 = 1540$, $C_9 = 1292$, $C_{10} = 1138$

- $v_j$ = value per unit of genre $j$:

  - Racing: $v_{\text{Racing}} = 28$
  - Sports: $v_{\text{Sports}} = 69$
  - Action: $v_{\text{Action}} = 20$
  - Adventure: $v_{\text{Adventure}} = 62$
  - RPG: $v_{\text{RPG}} = 58$
  - Shooter: $v_{\text{Shooter}} = 11$
  - Strategy: $v_{\text{Strategy}} = 73$
  - Simulation: $v_{\text{Simulation}} = 43$
  - Puzzle: $v_{\text{Puzzle}} = 28$
  - Fighting: $v_{\text{Fighting}} = 57$
  - Platformer: $v_{\text{Platformer}} = 92$
  - Survival: $v_{\text{Survival}} = 66$
  - Horror: $v_{\text{Horror}} = 14$
  - Sandbox: $v_{\text{Sandbox}} = 49$
  - MMO: $v_{\text{MMO}} = 12$

- $w_j$ = memory requirement per unit of genre $j$:

  - Racing: $w_{\text{Racing}} = 393$
  - Sports: $w_{\text{Sports}} = 195$
  - Action: $w_{\text{Action}} = 192$
  - Adventure: $w_{\text{Adventure}} = 155$
  - RPG: $w_{\text{RPG}} = 500$
  - Shooter: $w_{\text{Shooter}} = 156$
  - Strategy: $w_{\text{Strategy}} = 317$
  - Simulation: $w_{\text{Simulation}} = 694$
  - Puzzle: $w_{\text{Puzzle}} = 751$
  - Fighting: $w_{\text{Fighting}} = 467$
  - Platformer: $w_{\text{Platformer}} = 796$
  - Survival: $w_{\text{Survival}} = 146$
  - Horror: $w_{\text{Horror}} = 269$
  - Sandbox: $w_{\text{Sandbox}} = 246$
  - MMO: $w_{\text{MMO}} = 652$

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of games from genre $j$ to be listed on platform $i$

##### Objective Function

$\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}$

##### Constraints

1. **Platform Capacity Constraints:**

   For each platform $i \in I$:

   $\sum_{j \in J} w_j \, x_{ij} \leq C_i$

2. **Non-negativity and Integrality:**

   $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in I$, $j \in J$

##### Retrieved Information

{
  "platforms": {
    "1": 1336,
    "2": 1754,
    "3": 1617,
    "4": 1119,
    "5": 1410,
    "6": 627,
    "7": 748,
    "8": 1540,
    "9": 1292,
    "10": 1138
  },
  "genres": {
    "Racing": {"Value": 28, "Weight": 393},
    "Sports": {"Value": 69, "Weight": 195},
    "Action": {"Value": 20, "Weight": 192},
    "Adventure": {"Value": 62, "Weight": 155},
    "RPG": {"Value": 58, "Weight": 500},
    "Shooter": {"Value": 11, "Weight": 156},
    "Strategy": {"Value": 73, "Weight": 317},
    "Simulation": {"Value": 43, "Weight": 694},
    "Puzzle": {"Value": 28, "Weight": 751},
    "Fighting": {"Value": 57, "Weight": 467},
    "Platformer": {"Value": 92, "Weight": 796},
    "Survival": {"Value": 66, "Weight": 146},
    "Horror": {"Value": 14, "Weight": 269},
    "Sandbox": {"Value": 49, "Weight": 246},
    "MMO": {"Value": 12, "Weight": 652}
  }
}