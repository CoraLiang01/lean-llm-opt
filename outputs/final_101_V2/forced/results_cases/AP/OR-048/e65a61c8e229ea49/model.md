##### Objective Function:

$\quad \max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij}$

where $x_{ij}$ is the number of units of air conditioner type $j$ placed in storage area $i$, and $v_j$ is the value of product $j$.

##### Constraints

###### 1. Capacity Constraints (for each storage area):

$\sum_{j=1}^{10} w_j x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,15\}$

where $w_j$ is the weight (size) of product $j$, and $C_i$ is the capacity of storage area $i$.

###### 2. Variable Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,15\},\ j \in \{1,2,\ldots,10\}$

##### Retrieved Information

{
  "storage_areas": [
    {"StorageID": "1", "Capacity": 1083},
    {"StorageID": "2", "Capacity": 1840},
    {"StorageID": "3", "Capacity": 770},
    {"StorageID": "4", "Capacity": 1299},
    {"StorageID": "5", "Capacity": 1259},
    {"StorageID": "6", "Capacity": 543},
    {"StorageID": "7", "Capacity": 1831},
    {"StorageID": "8", "Capacity": 855},
    {"StorageID": "9", "Capacity": 619},
    {"StorageID": "10", "Capacity": 637},
    {"StorageID": "11", "Capacity": 935},
    {"StorageID": "12", "Capacity": 626},
    {"StorageID": "13", "Capacity": 1457},
    {"StorageID": "14", "Capacity": 1198},
    {"StorageID": "15", "Capacity": 837}
  ],
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

##### Parameter Definitions

Let $i \in \{1,2,\ldots,15\}$ index storage areas, and $j \in \{1,2,\ldots,10\}$ index air conditioner types.

- $C_i$ (storage area capacities):

  $C_1 = 1083,\ C_2 = 1840,\ C_3 = 770,\ C_4 = 1299,\ C_5 = 1259,\ C_6 = 543,\ C_7 = 1831,\ C_8 = 855,\ C_9 = 619,\ C_{10} = 637,\ C_{11} = 935,\ C_{12} = 626,\ C_{13} = 1457,\ C_{14} = 1198,\ C_{15} = 837$

- $v_j$ (product values):

  $v_1 = 4811$ (Window Unit), $v_2 = 1130$ (Portable Unit), $v_3 = 1611$ (Split System), $v_4 = 3368$ (Ductless System), $v_5 = 2135$ (Central AC), $v_6 = 1046$ (Hybrid AC), $v_7 = 4030$ (Geothermal AC), $v_8 = 3761$ (Smart AC), $v_9 = 3523$ (Evaporative Cooler), $v_{10} = 1701$ (Package Unit)

- $w_j$ (product weights):

  $w_1 = 114$ (Window Unit), $w_2 = 200$ (Portable Unit), $w_3 = 106$ (Split System), $w_4 = 256$ (Ductless System), $w_5 = 268$ (Central AC), $w_6 = 185$ (Hybrid AC), $w_7 = 299$ (Geothermal AC), $w_8 = 131$ (Smart AC), $w_9 = 139$ (Evaporative Cooler), $w_{10} = 105$ (Package Unit)

##### Decision Variables

$x_{ij}$: Number of units of air conditioner type $j$ placed in storage area $i$, integer and non-negative.