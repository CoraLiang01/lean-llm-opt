##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

##### Constraints

###### 1. Shelf Capacity Constraints:

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

###### 2. Non-negativity and Integrality Constraints:

$x_{ij} \geq 0$ and integer, $\quad \forall i \in \{1,2,\ldots,10\}, \; j \in \{1,2,\ldots,20\}$

##### Retrieved Information

{
  "shelves": {
    "1": 750,
    "2": 820,
    "3": 570,
    "4": 800,
    "5": 550,
    "6": 900,
    "7": 650,
    "8": 800,
    "9": 850,
    "10": 900
  },
  "products": {
    "1": {"Value": 55, "Weight": 10},
    "2": {"Value": 75, "Weight": 20},
    "3": {"Value": 65, "Weight": 5},
    "4": {"Value": 60, "Weight": 15},
    "5": {"Value": 80, "Weight": 25},
    "6": {"Value": 90, "Weight": 35},
    "7": {"Value": 40, "Weight": 45},
    "8": {"Value": 100, "Weight": 55},
    "9": {"Value": 55, "Weight": 65},
    "10": {"Value": 75, "Weight": 20},
    "11": {"Value": 110, "Weight": 18},
    "12": {"Value": 50, "Weight": 28},
    "13": {"Value": 60, "Weight": 8},
    "14": {"Value": 120, "Weight": 28},
    "15": {"Value": 70, "Weight": 25},
    "16": {"Value": 110, "Weight": 40},
    "17": {"Value": 50, "Weight": 55},
    "18": {"Value": 60, "Weight": 70},
    "19": {"Value": 120, "Weight": 85},
    "20": {"Value": 100, "Weight": 100}
  }
}

Where:
- $C_i$ is the capacity of shelf $i$.
- $v_j$ is the value of product $j$.
- $w_j$ is the weight of product $j$.
- $x_{ij}$ is the integer number of units of product $j$ placed on shelf $i$.