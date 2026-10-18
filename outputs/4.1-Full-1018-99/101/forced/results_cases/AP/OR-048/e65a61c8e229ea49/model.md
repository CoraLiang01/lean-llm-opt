##### Decision Variables

Let $x_{ij}$ be the number of units of air conditioner type $j$ placed in storage area $i$, where $i \in \{1,2,\ldots,15\}$ and $j \in \{1,2,\ldots,10\}$. All $x_{ij}$ are integer variables.

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

| $j$ | Product Name           | Value $v_j$ | Weight $w_j$ |
|-----|-----------------------|-------------|--------------|
| 1   | Window Unit           | 4811        | 114          |
| 2   | Portable Unit         | 1130        | 200          |
| 3   | Split System          | 1611        | 106          |
| 4   | Ductless System       | 3368        | 256          |
| 5   | Central AC            | 2135        | 268          |
| 6   | Hybrid AC             | 1046        | 185          |
| 7   | Geothermal AC         | 4030        | 299          |
| 8   | Smart AC              | 3761        | 131          |
| 9   | Evaporative Cooler    | 3523        | 139          |
| 10  | Package Unit          | 1701        | 105          |

##### Objective Function

$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

##### Constraints

For each storage area $i = 1, \ldots, 15$:

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i
$$

For all $i = 1, \ldots, 15$, $j = 1, \ldots, 10$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

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