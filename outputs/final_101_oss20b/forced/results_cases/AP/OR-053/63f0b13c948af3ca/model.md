##### Objective Function: 

$ \quad \quad \max \sum_{i \in \{1,\dots ,10\}} \sum_{j \in \{1,\dots ,20\}} v_{j}\,x_{ij} $

##### Constraints

###### 1. Shelf Capacity Constraints: 

$ \sum_{j=1}^{20} w_{j}\,x_{ij} \le C_{i} \quad \forall i \in \{1,\dots ,10\} $

###### 2. Variable Constraints: 

$ x_{ij} \in \mathbb{Z}_{\ge 0}, \quad \forall i \in \{1,\dots ,10\},\, j \in \{1,\dots ,20\} $

##### Retrieved Information

{
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
  "products": {
    "1": { "value": 50, "weight": 10 },
    "2": { "value": 70, "weight": 20 },
    "3": { "value": 30, "weight": 5 },
    "4": { "value": 60, "weight": 15 },
    "5": { "value": 80, "weight": 25 },
    "6": { "value": 90, "weight": 30 },
    "7": { "value": 40, "weight": 12 },
    "8": { "value": 100, "weight": 35 },
    "9": { "value": 55, "weight": 10 },
    "10": { "value": 75, "weight": 20 },
    "11": { "value": 65, "weight": 18 },
    "12": { "value": 95, "weight": 28 },
    "13": { "value": 45, "weight": 8 },
    "14": { "value": 85, "weight": 22 },
    "15": { "value": 70, "weight": 25 },
    "16": { "value": 110, "weight": 40 },
    "17": { "value": 50, "weight": 14 },
    "18": { "value": 60, "weight": 16 },
    "19": { "value": 120, "weight": 50 },
    "20": { "value": 100, "weight": 30 }
  },
  "shelves": [ "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" ],
  "products_list": [ "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20" ]
}