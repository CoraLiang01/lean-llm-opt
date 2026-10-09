##### Decision Variables

Let $x_{ij}$ denote the number of units of air conditioner type $j$ placed in storage area $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

##### Parameters

- Storage Areas and Capacities:
  - Storage Area 1: $C_1 = 1083$
  - Storage Area 2: $C_2 = 1840$
  - Storage Area 3: $C_3 = 770$
  - Storage Area 4: $C_4 = 1299$
  - Storage Area 5: $C_5 = 1259$
  - Storage Area 6: $C_6 = 543$
  - Storage Area 7: $C_7 = 1831$
  - Storage Area 8: $C_8 = 855$
  - Storage Area 9: $C_9 = 619$
  - Storage Area 10: $C_{10} = 637$
  - Storage Area 11: $C_{11} = 935$
  - Storage Area 12: $C_{12} = 626$
  - Storage Area 13: $C_{13} = 1457$
  - Storage Area 14: $C_{14} = 1198$
  - Storage Area 15: $C_{15} = 837$

- Air Conditioner Types, Values, and Weights:
  - 1: Window Unit, $v_1 = 4811$, $w_1 = 114$
  - 2: Portable Unit, $v_2 = 1130$, $w_2 = 200$
  - 3: Split System, $v_3 = 1611$, $w_3 = 106$
  - 4: Ductless System, $v_4 = 3368$, $w_4 = 256$
  - 5: Central AC, $v_5 = 2135$, $w_5 = 268$
  - 6: Hybrid AC, $v_6 = 1046$, $w_6 = 185$
  - 7: Geothermal AC, $v_7 = 4030$, $w_7 = 299$
  - 8: Smart AC, $v_8 = 3761$, $w_8 = 131$
  - 9: Evaporative Cooler, $v_9 = 3523$, $w_9 = 139$
  - 10: Package Unit, $v_{10} = 1701$, $w_{10} = 105$

##### Objective Function

$\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \, x_{ij}$

##### Constraints

For each storage area $i = 1, \ldots, 15$:

$\sum_{j=1}^{10} w_j \, x_{ij} \leq C_i$

For all $i = 1, \ldots, 15$, $j = 1, \ldots, 10$:

$x_{ij} \in \mathbb{Z}_{\geq 0}$

##### Retrieved Information

{
  "storage_areas": {
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
  "products": [
    {"ProductName": "Window Unit", "Value": 4811, "Weight": 114},
    {"ProductName": "Portable Unit", "Value": 1130, "Weight": 200},
    {"ProductName": "Split System", "Value": 1611, "Weight": 106},
    {"ProductName": "Ductless System", "Value": 3368, "Weight": 256},
    {"ProductName": "Central AC", "Value": 2135, "Weight": 268},
    {"ProductName": "Hybrid AC", "Value": 1046, "Weight": 185},
    {"ProductName": "Geothermal AC", "Value": 4030, "Weight": 299},
    {"ProductName": "Smart AC", "Value": 3761, "Weight": 131},
    {"ProductName": "Evaporative Cooler", "Value": 3523, "Weight": 139},
    {"ProductName": "Package Unit", "Value": 1701, "Weight": 105}
  ]
}