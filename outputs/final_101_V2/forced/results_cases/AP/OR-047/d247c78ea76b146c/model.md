##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \, x_{ij}$

where:
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$
- $v_j$ = value of one unit of genre $j$

##### Constraints

###### 1. Platform Capacity Constraints:

$\sum_{j=1}^{15} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

where:
- $w_j$ = memory requirement (weight) of one unit of genre $j$
- $C_i$ = memory capacity of platform $i$

###### 2. Variable Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}, \; j \in \{1,2,\ldots,15\}$

##### Retrieved Information

{
  "platforms": [
    {"PlatformId": "1", "Capacity": 1336},
    {"PlatformId": "2", "Capacity": 1754},
    {"PlatformId": "3", "Capacity": 1617},
    {"PlatformId": "4", "Capacity": 1119},
    {"PlatformId": "5", "Capacity": 1410},
    {"PlatformId": "6", "Capacity": 627},
    {"PlatformId": "7", "Capacity": 748},
    {"PlatformId": "8", "Capacity": 1540},
    {"PlatformId": "9", "Capacity": 1292},
    {"PlatformId": "10", "Capacity": 1138}
  ],
  "genres": [
    {"ProductName": "Racing", "Value": 28, "Weight": 393},
    {"ProductName": "Sports", "Value": 69, "Weight": 195},
    {"ProductName": "Action", "Value": 20, "Weight": 192},
    {"ProductName": "Adventure", "Value": 62, "Weight": 155},
    {"ProductName": "RPG", "Value": 58, "Weight": 500},
    {"ProductName": "Shooter", "Value": 11, "Weight": 156},
    {"ProductName": "Strategy", "Value": 73, "Weight": 317},
    {"ProductName": "Simulation", "Value": 43, "Weight": 694},
    {"ProductName": "Puzzle", "Value": 28, "Weight": 751},
    {"ProductName": "Fighting", "Value": 57, "Weight": 467},
    {"ProductName": "Platformer", "Value": 92, "Weight": 796},
    {"ProductName": "Survival", "Value": 66, "Weight": 146},
    {"ProductName": "Horror", "Value": 14, "Weight": 269},
    {"ProductName": "Sandbox", "Value": 49, "Weight": 246},
    {"ProductName": "MMO", "Value": 12, "Weight": 652}
  ]
}

##### Full Parameter List

- Platforms ($i$): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Genres ($j$): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO

- Platform Capacities ($C_i$):

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

- Genre Values ($v_j$) and Weights ($w_j$):

  - Racing: $v_1 = 28$, $w_1 = 393$
  - Sports: $v_2 = 69$, $w_2 = 195$
  - Action: $v_3 = 20$, $w_3 = 192$
  - Adventure: $v_4 = 62$, $w_4 = 155$
  - RPG: $v_5 = 58$, $w_5 = 500$
  - Shooter: $v_6 = 11$, $w_6 = 156$
  - Strategy: $v_7 = 73$, $w_7 = 317$
  - Simulation: $v_8 = 43$, $w_8 = 694$
  - Puzzle: $v_9 = 28$, $w_9 = 751$
  - Fighting: $v_{10} = 57$, $w_{10} = 467$
  - Platformer: $v_{11} = 92$, $w_{11} = 796$
  - Survival: $v_{12} = 66$, $w_{12} = 146$
  - Horror: $v_{13} = 14$, $w_{13} = 269$
  - Sandbox: $v_{14} = 49$, $w_{14} = 246$
  - MMO: $v_{15} = 12$, $w_{15} = 652$

##### Decision Variables

$x_{ij}$: integer, number of units of genre $j$ to be listed on platform $i$, for all $i \in \{1,\ldots,10\}$ and $j \in \{1,\ldots,15\}$