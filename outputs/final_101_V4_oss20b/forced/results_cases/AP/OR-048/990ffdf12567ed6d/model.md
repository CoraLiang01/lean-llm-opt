##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{15} \sum_{j=1}^{10} v_{j} x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\sum_{j=1}^{10} w_{j} x_{ij} \leq C_{i} \quad \forall i \in \{1,2,\dots ,15\}$

###### 2. Variable Constraints: 

$x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\dots ,15\}, \; \forall j \in \{1,\dots ,10\}$

###### Retrieved Information

{
  "capacity": {
    "1": 1083,
    "2": 1840,
    "3": 770,
    "4": 1299,
    "5": 1259,
    "6": 543,
    "7": 1831,
    "8": 855,
    "9": 619,
    "10": 637,
    "11": 935,
    "12": 626,
    "13": 1457,
    "14": 1198,
    "15": 837
  },
  "products": {
    "Window Unit": {
      "Value": 4811,
      "Weight": 114
    },
    "Portable Unit": {
      "Value": 1130,
      "Weight": 200
    },
    "Split System": {
      "Value": 1611,
      "Weight": 106
    },
    "Ductless System": {
      "Value": 3368,
      "Weight": 256
    },
    "Central AC": {
      "Value": 2135,
      "Weight": 268
    },
    "Hybrid AC": {
      "Value": 1046,
      "Weight": 185
    },
    "Geothermal AC": {
      "Value": 4030,
      "Weight": 299
    },
    "Smart AC": {
      "Value": 3761,
      "Weight": 131
    },
    "Evaporative Cooler": {
      "Value": 3523,
      "Weight": 139
    },
    "Package Unit": {
      "Value": 1701,
      "Weight": 105
    }
  },
  "storage_areas": [
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
  ],
  "products_list": [
    "Window Unit",
    "Portable Unit",
    "Split System",
    "Ductless System",
    "Central AC",
    "Hybrid AC",
    "Geothermal AC",
    "Smart AC",
    "Evaporative Cooler",
    "Package Unit"
  ]
}