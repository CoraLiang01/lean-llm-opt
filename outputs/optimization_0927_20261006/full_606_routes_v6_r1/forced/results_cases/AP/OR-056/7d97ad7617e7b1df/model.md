##### Objective Function:

$\quad \max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the number of vessels of type $j$ placed in display area $i$, $v_j$ is the value of vessel type $j$.

##### Constraints:

###### 1. Capacity Constraints (for each display area $i$):

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,14\}$

where $w_j$ is the weight (size) of vessel type $j$, $C_i$ is the capacity of display area $i$.

###### 2. Non-negativity and Integrality:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,14\}, \; j \in \{1,\ldots,20\}$

---

##### Retrieved Information

```json
{
  "display_areas": [
    {"DisplayID": "1", "Capacity": 457},
    {"DisplayID": "2", "Capacity": 604},
    {"DisplayID": "3", "Capacity": 751},
    {"DisplayID": "4", "Capacity": 468},
    {"DisplayID": "5", "Capacity": 343},
    {"DisplayID": "6", "Capacity": 408},
    {"DisplayID": "7", "Capacity": 741},
    {"DisplayID": "8", "Capacity": 914},
    {"DisplayID": "9", "Capacity": 682},
    {"DisplayID": "10", "Capacity": 409},
    {"DisplayID": "11", "Capacity": 342},
    {"DisplayID": "12", "Capacity": 903},
    {"DisplayID": "13", "Capacity": 680},
    {"DisplayID": "14", "Capacity": 886}
  ],
  "products": [
    {"ProductName": "Speedboat", "Value": 29664, "Weight": 18},
    {"ProductName": "Fishing Boat", "Value": 31778, "Weight": 36},
    {"ProductName": "Catamaran", "Value": 73501, "Weight": 25},
    {"ProductName": "Yacht", "Value": 78255, "Weight": 16},
    {"ProductName": "Sailboat", "Value": 93606, "Weight": 97},
    {"ProductName": "Kayak", "Value": 46983, "Weight": 35},
    {"ProductName": "Canoe", "Value": 95026, "Weight": 32},
    {"ProductName": "Houseboat", "Value": 57685, "Weight": 100},
    {"ProductName": "Pontoon", "Value": 60323, "Weight": 43},
    {"ProductName": "Jet Ski", "Value": 91224, "Weight": 15},
    {"ProductName": "Rowboat", "Value": 44003, "Weight": 95},
    {"ProductName": "Hovercraft", "Value": 75998, "Weight": 57},
    {"ProductName": "Cabin Cruiser", "Value": 84525, "Weight": 13},
    {"ProductName": "Wakeboard Boat", "Value": 66207, "Weight": 44},
    {"ProductName": "Dinghy", "Value": 65002, "Weight": 64},
    {"ProductName": "Trawler", "Value": 33132, "Weight": 88},
    {"ProductName": "Paddle Boat", "Value": 69239, "Weight": 42},
    {"ProductName": "Submarine", "Value": 66948, "Weight": 46},
    {"ProductName": "RIB", "Value": 88240, "Weight": 24},
    {"ProductName": "Skiff", "Value": 48858, "Weight": 93}
  ]
}
```

- Display areas $i \in \{1,2,\ldots,14\}$ with capacities $C_i$ as listed above.
- Vessel types $j \in \{1,2,\ldots,20\}$ with values $v_j$ and weights $w_j$ as listed above.
- Decision variables $x_{ij}$: number of vessels of type $j$ in display area $i$.

##### Full Parameter Tables

| DisplayID | Capacity |
|-----------|----------|
| 1         | 457      |
| 2         | 604      |
| 3         | 751      |
| 4         | 468      |
| 5         | 343      |
| 6         | 408      |
| 7         | 741      |
| 8         | 914      |
| 9         | 682      |
| 10        | 409      |
| 11        | 342      |
| 12        | 903      |
| 13        | 680      |
| 14        | 886      |

| $j$ | ProductName      | Value  $v_j$ | Weight $w_j$ |
|-----|------------------|-------------|--------------|
| 1   | Speedboat        | 29664       | 18           |
| 2   | Fishing Boat     | 31778       | 36           |
| 3   | Catamaran        | 73501       | 25           |
| 4   | Yacht            | 78255       | 16           |
| 5   | Sailboat         | 93606       | 97           |
| 6   | Kayak            | 46983       | 35           |
| 7   | Canoe            | 95026       | 32           |
| 8   | Houseboat        | 57685       | 100          |
| 9   | Pontoon          | 60323       | 43           |
| 10  | Jet Ski          | 91224       | 15           |
| 11  | Rowboat          | 44003       | 95           |
| 12  | Hovercraft       | 75998       | 57           |
| 13  | Cabin Cruiser    | 84525       | 13           |
| 14  | Wakeboard Boat   | 66207       | 44           |
| 15  | Dinghy           | 65002       | 64           |
| 16  | Trawler          | 33132       | 88           |
| 17  | Paddle Boat      | 69239       | 42           |
| 18  | Submarine        | 66948       | 46           |
| 19  | RIB              | 88240       | 24           |
| 20  | Skiff            | 48858       | 93           |