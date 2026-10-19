##### Objective Function: 

$ \quad \quad \max \sum_{i} \sum_{j} p_j \cdot x_{ij} $

##### Constraints

###### 1. Capacity Constraints: 

$ \quad \quad \sum_{j} w_j \, x_{ij} \leq c_i , \quad \forall i $

###### 2. Non-negativity and Integrality Constraints: 

$ \quad \quad x_{ij} \in \mathbb{Z}_{\geq 0} , \quad \forall i,j $

The indices $i$ run over the shelves, $j$ run over the products. $p_j$ is the value of product $j$, $w_j$ is the weight of product $j$, and $c_i$ is the capacity of shelf $i$.

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
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20"
  ],
  "profit": {
    "1": 50,
    "2": 70,
    "3": 30,
    "4": 60,
    "5": 80,
    "6": 90,
    "7": 40,
    "8": 100,
    "9": 55,
    "10": 75,
    "11": 65,
    "12": 95,
    "13": 45,
    "14": 85,
    "15": 70,
    "16": 110,
    "17": 50,
    "18": 60,
    "19": 120,
    "20": 100
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
  "capacity": {
    "1": 500,
    "2": 700,
    "3": 600,
    "4": 800,
    "5": 550,
    "6": 900,
    "7": 650,
    "8": 750,
    "9": 820,
    "10": 570
  },
  "weight": {
    "1": 10,
    "2": 20,
    "3": 5,
    "4": 15,
    "5": 25,
    "6": 30,
    "7": 12,
    "8": 35,
    "9": 10,
    "10": 20,
    "11": 18,
    "12": 28,
    "13": 8,
    "14": 22,
    "15": 25,
    "16": 40,
    "17": 14,
    "18": 16,
    "19": 50,
    "20": 30
  }
}