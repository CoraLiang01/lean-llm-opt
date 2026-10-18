##### Objective Function:

$\quad \max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the integer number of units of boat type $j$ placed in display area $i$, and $v_j$ is the value of boat type $j$.

##### Constraints:

###### 1. Capacity Constraints (for each display area $i$):

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,14\}$

where $w_j$ is the weight (size) of boat type $j$, and $C_i$ is the capacity of display area $i$.

###### 2. Nonnegativity and Integrality:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,14\}, \; j \in \{1,2,\ldots,20\}$

---

##### Retrieved Information

```json
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
```

- Display areas: 14, with capacities as listed above.
- Boat types: 20, with names, values, and weights as listed above.
- Decision variables: $x_{ij}$ = number of units of boat type $j$ in display area $i$, integer and nonnegative.

##### Full Model (with indices):

Let $I = \{1,2,\ldots,14\}$ (display areas), $J = \{1,2,\ldots,20\}$ (boat types).

$\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

subject to

$\sum_{j \in J} w_j x_{ij} \leq C_i \quad \forall i \in I$

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J$

where

- $C_i$ (capacity for display area $i$): [356, 478, 305, 291, 168, 449, 139, 383, 472, 288, 320, 250, 402, 293]
- $v_j$ (value for boat type $j$): [69978, 54011, 36352, 51521, 50415, 76109, 50462, 28989, 23318, 26142, 42040, 85961, 50142, 48478, 60953, 95265, 22839, 90957, 84652, 78991]
- $w_j$ (weight for boat type $j$): [18, 42, 49, 42, 41, 48, 22, 29, 45, 14, 38, 47, 45, 28, 24, 39, 32, 36, 14, 16]