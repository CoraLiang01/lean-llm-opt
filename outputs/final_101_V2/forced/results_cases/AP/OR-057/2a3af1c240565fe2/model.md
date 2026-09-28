##### Decision Variables

Let $x_{ij}$ be the number of units of game $j$ (genre) to be listed on platform $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Let $i \in \{1,2,\ldots,15\}$ index the platforms, and $j \in \{1,2,\ldots,15\}$ index the games/genres.

##### Parameters

- Platform capacities:
  - $C_1 = 995$
  - $C_2 = 1143$
  - $C_3 = 949$
  - $C_4 = 969$
  - $C_5 = 1649$
  - $C_6 = 870$
  - $C_7 = 1064$
  - $C_8 = 536$
  - $C_9 = 766$
  - $C_{10} = 532$
  - $C_{11} = 1703$
  - $C_{12} = 1633$
  - $C_{13} = 1203$
  - $C_{14} = 1979$
  - $C_{15} = 1797$

- Game values and memory requirements:

| $j$ | ProductName   | $v_j$ (Value) | $w_j$ (Weight) |
|-----|--------------|---------------|---------------|
| 1   | Racing       | 59            | 776           |
| 2   | Sports       | 83            | 573           |
| 3   | Action       | 94            | 127           |
| 4   | Adventure    | 41            | 138           |
| 5   | RPG          | 96            | 385           |
| 6   | Shooter      | 12            | 263           |
| 7   | Strategy     | 83            | 473           |
| 8   | Simulation   | 36            | 387           |
| 9   | Puzzle       | 56            | 390           |
| 10  | Fighting     | 27            | 556           |
| 11  | Platformer   | 47            | 601           |
| 12  | Survival     | 24            | 441           |
| 13  | Horror       | 14            | 603           |
| 14  | Sandbox      | 22            | 411           |
| 15  | MMO          | 17            | 652           |

##### Objective Function

$\max \sum_{i=1}^{15} \sum_{j=1}^{15} v_j \, x_{ij}$

##### Constraints

###### 1. Platform Memory Capacity Constraints

For each platform $i$:

$\sum_{j=1}^{15} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,15\}$

###### 2. Integer and Non-negativity Constraints

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,15\}, \; j \in \{1,2,\ldots,15\}$

##### Retrieved Information

{
  "platforms": {
    "1": 995,
    "2": 1143,
    "3": 949,
    "4": 969,
    "5": 1649,
    "6": 870,
    "7": 1064,
    "8": 536,
    "9": 766,
    "10": 532,
    "11": 1703,
    "12": 1633,
    "13": 1203,
    "14": 1979,
    "15": 1797
  },
  "products": [
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