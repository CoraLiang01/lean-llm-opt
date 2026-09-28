##### Objective Function:

$\quad \max \sum_{i=1}^{14} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the number of vessels of type $j$ placed in display area $i$, $v_j$ is the value of vessel type $j$.

##### Constraints

###### 1. Capacity Constraints (for each display area $i$):

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,14\}$

where $w_j$ is the weight (size) of vessel type $j$, $C_i$ is the capacity of display area $i$.

###### 2. Variable Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,14\}, \; j \in \{1,\ldots,20\}$

##### Retrieved Information

{
  "display_areas": {
    "1": 457,
    "2": 604,
    "3": 751,
    "4": 468,
    "5": 343,
    "6": 408,
    "7": 741,
    "8": 914,
    "9": 682,
    "10": 409,
    "11": 342,
    "12": 903,
    "13": 680,
    "14": 886
  },
  "vessel_types": [
    {
      "ProductName": "Speedboat",
      "Value": 29664,
      "Weight": 18
    },
    {
      "ProductName": "Fishing Boat",
      "Value": 31778,
      "Weight": 36
    },
    {
      "ProductName": "Catamaran",
      "Value": 73501,
      "Weight": 25
    },
    {
      "ProductName": "Yacht",
      "Value": 78255,
      "Weight": 16
    },
    {
      "ProductName": "Sailboat",
      "Value": 93606,
      "Weight": 97
    },
    {
      "ProductName": "Kayak",
      "Value": 46983,
      "Weight": 35
    },
    {
      "ProductName": "Canoe",
      "Value": 95026,
      "Weight": 32
    },
    {
      "ProductName": "Houseboat",
      "Value": 57685,
      "Weight": 100
    },
    {
      "ProductName": "Pontoon",
      "Value": 60323,
      "Weight": 43
    },
    {
      "ProductName": "Jet Ski",
      "Value": 91224,
      "Weight": 15
    },
    {
      "ProductName": "Rowboat",
      "Value": 44003,
      "Weight": 95
    },
    {
      "ProductName": "Hovercraft",
      "Value": 75998,
      "Weight": 57
    },
    {
      "ProductName": "Cabin Cruiser",
      "Value": 84525,
      "Weight": 13
    },
    {
      "ProductName": "Wakeboard Boat",
      "Value": 66207,
      "Weight": 44
    },
    {
      "ProductName": "Dinghy",
      "Value": 65002,
      "Weight": 64
    },
    {
      "ProductName": "Trawler",
      "Value": 33132,
      "Weight": 88
    },
    {
      "ProductName": "Paddle Boat",
      "Value": 69239,
      "Weight": 42
    },
    {
      "ProductName": "Submarine",
      "Value": 66948,
      "Weight": 46
    },
    {
      "ProductName": "RIB",
      "Value": 88240,
      "Weight": 24
    },
    {
      "ProductName": "Skiff",
      "Value": 48858,
      "Weight": 93
    }
  ]
}

##### Parameter Definitions

- $C_i$: Capacity of display area $i$, as given above.
- $v_j$: Value of vessel type $j$, as given above.
- $w_j$: Weight (size) of vessel type $j$, as given above.
- $x_{ij}$: Number of vessels of type $j$ placed in display area $i$, integer and non-negative.