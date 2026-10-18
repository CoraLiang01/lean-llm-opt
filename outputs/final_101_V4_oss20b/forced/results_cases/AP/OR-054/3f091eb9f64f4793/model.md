##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_{j}\,x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\sum_{j=1}^{20} w_{j}\,x_{ij} \le C_{i} \quad \forall i \in \{1,2,\dots,10\}$

###### 2. Variable Constraints: 

$x_{ij} \in \mathbb{Z}_{\ge 0}, \quad \forall i,j$

###### Retrieved Information

{
  "capacity": {
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
    "1": {"value": 55, "weight": 10},
    "2": {"value": 75, "weight": 20},
    "3": {"value": 65, "weight": 5},
    "4": {"value": 60, "weight": 15},
    "5": {"value": 80, "weight": 25},
    "6": {"value": 90, "weight": 35},
    "7": {"value": 40, "weight": 45},
    "8": {"value": 100, "weight": 55},
    "9": {"value": 55, "weight": 65},
    "10": {"value": 75, "weight": 20},
    "11": {"value": 110, "weight": 18},
    "12": {"value": 50,  "weight": 28},
    "13": {"value": 60,  "weight": 8},
    "14": {"value": 120, "weight": 28},
    "15": {"value": 70,  "weight": 25},
    "16": {"value": 110, "weight": 40},
    "17": {"value": 50,  "weight": 55},
    "18": {"value": 60,  "weight": 70},
    "19": {"value": 120, "weight": 85},
    "20": {"value": 100, "weight": 100}
  },
  "shelves": ["1","2","3","4","5","6","7","8","9","10"],
  "products_list": ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
}