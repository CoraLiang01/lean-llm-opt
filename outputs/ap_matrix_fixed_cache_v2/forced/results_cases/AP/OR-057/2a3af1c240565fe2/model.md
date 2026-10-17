##### Objective Function:

$\quad \max \sum_{i=1}^{15} \sum_{j=1}^{15} v_j \, x_{ij}$

where $x_{ij}$ is the integer number of units of game $j$ to be listed on platform $i$, and $v_j$ is the value of game $j$.

##### Constraints:

###### 1. Platform Memory Capacity Constraints:

$\sum_{j=1}^{15} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,15\}$

where $w_j$ is the memory requirement (weight) of game $j$, and $C_i$ is the memory capacity of platform $i$.

###### 2. Variable Domain Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,15\}, \; j \in \{1,2,\ldots,15\}$

##### Retrieved Information

{
  "platforms": [
    {"PlatformID": "1", "Capacity": 995},
    {"PlatformID": "2", "Capacity": 1143},
    {"PlatformID": "3", "Capacity": 949},
    {"PlatformID": "4", "Capacity": 969},
    {"PlatformID": "5", "Capacity": 1649},
    {"PlatformID": "6", "Capacity": 870},
    {"PlatformID": "7", "Capacity": 1064},
    {"PlatformID": "8", "Capacity": 536},
    {"PlatformID": "9", "Capacity": 766},
    {"PlatformID": "10", "Capacity": 532},
    {"PlatformID": "11", "Capacity": 1703},
    {"PlatformID": "12", "Capacity": 1633},
    {"PlatformID": "13", "Capacity": 1203},
    {"PlatformID": "14", "Capacity": 1979},
    {"PlatformID": "15", "Capacity": 1797}
  ],
  "games": [
    {"ProductName": "Racing", "Value": 59, "Weight": 776},
    {"ProductName": "Sports", "Value": 83, "Weight": 573},
    {"ProductName": "Action", "Value": 94, "Weight": 127},
    {"ProductName": "Adventure", "Value": 41, "Weight": 138},
    {"ProductName": "RPG", "Value": 96, "Weight": 385},
    {"ProductName": "Shooter", "Value": 12, "Weight": 263},
    {"ProductName": "Strategy", "Value": 83, "Weight": 473},
    {"ProductName": "Simulation", "Value": 36, "Weight": 387},
    {"ProductName": "Puzzle", "Value": 56, "Weight": 390},
    {"ProductName": "Fighting", "Value": 27, "Weight": 556},
    {"ProductName": "Platformer", "Value": 47, "Weight": 601},
    {"ProductName": "Survival", "Value": 24, "Weight": 441},
    {"ProductName": "Horror", "Value": 14, "Weight": 603},
    {"ProductName": "Sandbox", "Value": 22, "Weight": 411},
    {"ProductName": "MMO", "Value": 17, "Weight": 652}
  ]
}

##### Parameter Matrices

Let the set of platforms $I = \{1,2,\ldots,15\}$ and the set of games $J = \{1,2,\ldots,15\}$, with the following mappings:

- Platforms (with capacities $C_i$):

| $i$ | PlatformID | $C_i$ |
|-----|------------|-------|
| 1   | 1          | 995   |
| 2   | 2          | 1143  |
| 3   | 3          | 949   |
| 4   | 4          | 969   |
| 5   | 5          | 1649  |
| 6   | 6          | 870   |
| 7   | 7          | 1064  |
| 8   | 8          | 536   |
| 9   | 9          | 766   |
| 10  | 10         | 532   |
| 11  | 11         | 1703  |
| 12  | 12         | 1633  |
| 13  | 13         | 1203  |
| 14  | 14         | 1979  |
| 15  | 15         | 1797  |

- Games (with values $v_j$ and weights $w_j$):

| $j$ | ProductName  | $v_j$ | $w_j$ |
|-----|--------------|-------|-------|
| 1   | Racing       | 59    | 776   |
| 2   | Sports       | 83    | 573   |
| 3   | Action       | 94    | 127   |
| 4   | Adventure    | 41    | 138   |
| 5   | RPG          | 96    | 385   |
| 6   | Shooter      | 12    | 263   |
| 7   | Strategy     | 83    | 473   |
| 8   | Simulation   | 36    | 387   |
| 9   | Puzzle       | 56    | 390   |
| 10  | Fighting     | 27    | 556   |
| 11  | Platformer   | 47    | 601   |
| 12  | Survival     | 24    | 441   |
| 13  | Horror       | 14    | 603   |
| 14  | Sandbox      | 22    | 411   |
| 15  | MMO          | 17    | 652   |

##### Decision Variables

$x_{ij}$: integer, number of units of game $j$ to be listed on platform $i$, for all $i \in I$, $j \in J$.