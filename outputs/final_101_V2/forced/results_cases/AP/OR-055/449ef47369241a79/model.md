##### Objective Function:

$\quad \max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the integer number of units of boat type $j$ placed in display area $i$, $v_j$ is the value of boat type $j$.

##### Constraints

###### 1. Capacity Constraints (for each display area $i$):

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,14\}$

where $w_j$ is the weight (size) of boat type $j$, $C_i$ is the capacity of display area $i$.

###### 2. Variable Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,14\}, \; j \in \{1,\ldots,20\}$

##### Retrieved Information

{
  "display_areas": [
    {"DisplayID": "1", "Capacity": 356},
    {"DisplayID": "2", "Capacity": 478},
    {"DisplayID": "3", "Capacity": 305},
    {"DisplayID": "4", "Capacity": 291},
    {"DisplayID": "5", "Capacity": 168},
    {"DisplayID": "6", "Capacity": 449},
    {"DisplayID": "7", "Capacity": 139},
    {"DisplayID": "8", "Capacity": 383},
    {"DisplayID": "9", "Capacity": 472},
    {"DisplayID": "10", "Capacity": 288},
    {"DisplayID": "11", "Capacity": 320},
    {"DisplayID": "12", "Capacity": 250},
    {"DisplayID": "13", "Capacity": 402},
    {"DisplayID": "14", "Capacity": 293}
  ],
  "products": [
    {"ProductName": "Speedboat", "Value": 69978, "Weight": 18},
    {"ProductName": "Fishing Boat", "Value": 54011, "Weight": 42},
    {"ProductName": "Catamaran", "Value": 36352, "Weight": 49},
    {"ProductName": "Yacht", "Value": 51521, "Weight": 42},
    {"ProductName": "Sailboat", "Value": 50415, "Weight": 41},
    {"ProductName": "Kayak", "Value": 76109, "Weight": 48},
    {"ProductName": "Canoe", "Value": 50462, "Weight": 22},
    {"ProductName": "Houseboat", "Value": 28989, "Weight": 29},
    {"ProductName": "Pontoon", "Value": 23318, "Weight": 45},
    {"ProductName": "Jet Ski", "Value": 26142, "Weight": 14},
    {"ProductName": "Rowboat", "Value": 42040, "Weight": 38},
    {"ProductName": "Hovercraft", "Value": 85961, "Weight": 47},
    {"ProductName": "Cabin Cruiser", "Value": 50142, "Weight": 45},
    {"ProductName": "Wakeboard Boat", "Value": 48478, "Weight": 28},
    {"ProductName": "Dinghy", "Value": 60953, "Weight": 24},
    {"ProductName": "Trawler", "Value": 95265, "Weight": 39},
    {"ProductName": "Paddle Boat", "Value": 22839, "Weight": 32},
    {"ProductName": "Submarine", "Value": 90957, "Weight": 36},
    {"ProductName": "RIB", "Value": 84652, "Weight": 14},
    {"ProductName": "Skiff", "Value": 78991, "Weight": 16}
  ]
}

##### Parameter Definitions

- $C_i$: Capacity of display area $i$, for $i=1,\ldots,14$.
- $v_j$: Value of boat type $j$, for $j=1,\ldots,20$.
- $w_j$: Weight (size) of boat type $j$, for $j=1,\ldots,20$.
- $x_{ij}$: Integer number of units of boat type $j$ placed in display area $i$.

##### Explicit Parameter Values

- Display area capacities ($C_i$):

  - $C_1 = 356$
  - $C_2 = 478$
  - $C_3 = 305$
  - $C_4 = 291$
  - $C_5 = 168$
  - $C_6 = 449$
  - $C_7 = 139$
  - $C_8 = 383$
  - $C_9 = 472$
  - $C_{10} = 288$
  - $C_{11} = 320$
  - $C_{12} = 250$
  - $C_{13} = 402$
  - $C_{14} = 293$

- Boat types ($j=1,\ldots,20$):

| $j$ | Product Name      | $v_j$  | $w_j$ |
|-----|------------------|--------|-------|
| 1   | Speedboat        | 69978  | 18    |
| 2   | Fishing Boat     | 54011  | 42    |
| 3   | Catamaran        | 36352  | 49    |
| 4   | Yacht            | 51521  | 42    |
| 5   | Sailboat         | 50415  | 41    |
| 6   | Kayak            | 76109  | 48    |
| 7   | Canoe            | 50462  | 22    |
| 8   | Houseboat        | 28989  | 29    |
| 9   | Pontoon          | 23318  | 45    |
| 10  | Jet Ski          | 26142  | 14    |
| 11  | Rowboat          | 42040  | 38    |
| 12  | Hovercraft       | 85961  | 47    |
| 13  | Cabin Cruiser    | 50142  | 45    |
| 14  | Wakeboard Boat   | 48478  | 28    |
| 15  | Dinghy           | 60953  | 24    |
| 16  | Trawler          | 95265  | 39    |
| 17  | Paddle Boat      | 22839  | 32    |
| 18  | Submarine        | 90957  | 36    |
| 19  | RIB              | 84652  | 14    |
| 20  | Skiff            | 78991  | 16    |