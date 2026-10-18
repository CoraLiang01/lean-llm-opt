##### Objective Function: 

$\quad \quad \max \sum_{i} \sum_{j} p_j \cdot x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\quad \quad \sum_{j} a_j \, x_{ij} \leq c_i, \quad \forall i$

###### 2. Non-negativity and Integrality Constraints: 

$\quad \quad  x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i,j $

I indexes the ten products; J indexes the eight sections. $p_j$ is Value, $a_j$ is Weight, and $c_i$ is Capacity. Variables are nonnegative integers representing whole product units.

###### Retrieved Information

{
  "items": [
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
  "profit": {
    "1": 10,
    "2": 15,
    "3": 8,
    "4": 12,
    "5": 20,
    "6": 25,
    "7": 5,
    "8": 30,
    "9": 18,
    "10": 22
  },
  "shelves": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8"
  ],
  "space": {
    "1": 2,
    "2": 3,
    "3": 1,
    "4": 2,
    "5": 4,
    "6": 5,
    "7": 1,
    "8": 6,
    "9": 3,
    "10": 4
  }
}