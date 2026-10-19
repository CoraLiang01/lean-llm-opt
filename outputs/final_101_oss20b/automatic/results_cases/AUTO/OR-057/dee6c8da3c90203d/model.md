##### Objective Function: 

$$ \max \sum_{i \in I} \sum_{j \in J} p_j \cdot x_{ij} $$

##### Constraints

###### 1. Capacity Constraints: 

$$ \sum_{j \in J} w_j \, x_{ij} \leq c_i, \quad \forall i \in I $$

###### 2. Non-negativity and Integrality Constraints: 

$$ x_{ij} \in \mathbb{Z}_{\ge 0}, \quad \forall i \in I, \; j \in J $$

I indexes the 15 platforms; J indexes the 15 product genres.  
p_j is the product value, w_j is the memory requirement, and c_i is the platform capacity.  
Variables are nonnegative integers representing whole product units.

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
    "Racing": 59,
    "Sports": 83,
    "Action": 94,
    "Adventure": 41,
    "RPG": 96,
    "Shooter": 12,
    "Strategy": 83,
    "Simulation": 36,
    "Puzzle": 56,
    "Fighting": 27,
    "Platformer": 47,
    "Survival": 24,
    "Horror": 14,
    "Sandbox": 22,
    "MMO": 17
  },
  "space": {
    "Racing": 776,
    "Sports": 573,
    "Action": 127,
    "Adventure": 138,
    "RPG": 385,
    "Shooter": 263,
    "Strategy": 473,
    "Simulation": 387,
    "Puzzle": 390,
    "Fighting": 556,
    "Platformer": 601,
    "Survival": 441,
    "Horror": 603,
    "Sandbox": 411,
    "MMO": 652
  },
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
    "10",
    "11",
    "12",
    "13",
    "14",
    "15"
  ]
}