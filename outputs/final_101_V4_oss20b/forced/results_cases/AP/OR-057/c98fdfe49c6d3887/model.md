##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{15} \sum_{j=1}^{15} v_{j} \, x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\sum_{j=1}^{15} w_{j} \, x_{ij} \leq C_{i} \quad \forall i \in \{1,2,\dots ,15\}$

###### 2. Variable Constraints: 

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i,j$

###### Retrieved Information

{
  "capacity": {
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
  "products": {
    "Racing": {
      "value": 59,
      "weight": 776
    },
    "Sports": {
      "value": 83,
      "weight": 573
    },
    "Action": {
      "value": 94,
      "weight": 127
    },
    "Adventure": {
      "value": 41,
      "weight": 138
    },
    "RPG": {
      "value": 96,
      "weight": 385
    },
    "Shooter": {
      "value": 12,
      "weight": 263
    },
    "Strategy": {
      "value": 83,
      "weight": 473
    },
    "Simulation": {
      "value": 36,
      "weight": 387
    },
    "Puzzle": {
      "value": 56,
      "weight": 390
    },
    "Fighting": {
      "value": 27,
      "weight": 556
    },
    "Platformer": {
      "value": 47,
      "weight": 601
    },
    "Survival": {
      "value": 24,
      "weight": 441
    },
    "Horror": {
      "value": 14,
      "weight": 603
    },
    "Sandbox": {
      "value": 22,
      "weight": 411
    },
    "MMO": {
      "value": 17,
      "weight": 652
    }
  },
  "platforms": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15
  ],
  "products_list": [
    "Racing",
    "Sports",
    "Action",
    "Adventure",
    "RPG",
    "Shooter",
    "Strategy",
    "Simulation",
    "Puzzle",
    "Fighting",
    "Platformer",
    "Survival",
    "Horror",
    "Sandbox",
    "MMO"
  ]
}