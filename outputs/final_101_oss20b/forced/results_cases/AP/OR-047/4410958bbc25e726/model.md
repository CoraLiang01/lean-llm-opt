##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{10} \sum_{j=1}^{15} v_{j} x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\sum_{j=1}^{15} w_{j} x_{ij} \le C_{i} \quad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}$

###### 2. Variable Constraints: 

$ x_{ij} \in \mathbb{Z}_{\ge 0}, \quad \forall i \in \{1,\dots ,10\}, \; \forall j \in \{1,\dots ,15\}$

###### Retrieved Information

{
  "capacity": {
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
  "products": {
    "Racing": {
      "Value": 28,
      "Weight": 393
    },
    "Sports": {
      "Value": 69,
      "Weight": 195
    },
    "Action": {
      "Value": 20,
      "Weight": 192
    },
    "Adventure": {
      "Value": 62,
      "Weight": 155
    },
    "RPG": {
      "Value": 58,
      "Weight": 500
    },
    "Shooter": {
      "Value": 11,
      "Weight": 156
    },
    "Strategy": {
      "Value": 73,
      "Weight": 317
    },
    "Simulation": {
      "Value": 43,
      "Weight": 694
    },
    "Puzzle": {
      "Value": 28,
      "Weight": 751
    },
    "Fighting": {
      "Value": 57,
      "Weight": 467
    },
    "Platformer": {
      "Value": 92,
      "Weight": 796
    },
    "Survival": {
      "Value": 66,
      "Weight": 146
    },
    "Horror": {
      "Value": 14,
      "Weight": 269
    },
    "Sandbox": {
      "Value": 49,
      "Weight": 246
    },
    "MMO": {
      "Value": 12,
      "Weight": 652
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
    10
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