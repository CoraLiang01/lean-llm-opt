##### Objective Function: 

$ \quad \quad \max \sum_{i \in I} \sum_{j \in J} p_j \cdot x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$ \quad \quad \sum_{j \in J} a_j \, x_{ij} \le c_i, \quad \forall i \in I$

###### 2. Non-negativity and Integrality Constraints: 

$ \quad \quad x_{ij} \in \mathbb{Z}_{\ge 0}, \quad \forall i \in I, \forall j \in J$

I indexes the platforms (I = {1,2,3,4,5,6,7,8,9,10}); J indexes the product genres; $p_j$ is the value of product j; $a_j$ is the space requirement of product j; $c_i$ is the total capacity of platform i.

###### Retrieved Information

{
  "items": [
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
  ],
  "profit": {
    "Racing": 28,
    "Sports": 69,
    "Action": 20,
    "Adventure": 62,
    "RPG": 58,
    "Shooter": 11,
    "Strategy": 73,
    "Simulation": 43,
    "Puzzle": 28,
    "Fighting": 57,
    "Platformer": 92,
    "Survival": 66,
    "Horror": 14,
    "Sandbox": 49,
    "MMO": 12
  },
  "shelves": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10"
  ],
  "space": {
    "Racing": 393,
    "Sports": 195,
    "Action": 192,
    "Adventure": 155,
    "RPG": 500,
    "Shooter": 156,
    "Strategy": 317,
    "Simulation": 694,
    "Puzzle": 751,
    "Fighting": 467,
    "Platformer": 796,
    "Survival": 146,
    "Horror": 269,
    "Sandbox": 246,
    "MMO": 652
  },
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
  }
}