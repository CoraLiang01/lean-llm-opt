##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j \, x_{ij}$

where $x_{ij}$ is the number of units of product $j$ placed in cabinet $i$, $v_j$ is the value of product $j$.

##### Constraints

###### 1. Cabinet Capacity Constraints:

$\sum_{j=1}^{18} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

where $w_j$ is the weight of product $j$, $C_i$ is the capacity of cabinet $i$.

###### 2. Variable Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}, \forall j \in \{1,2,\ldots,18\}$

##### Retrieved Information

{
  "cabinets": [
    {"CabinetID": 1, "Capacity": 400},
    {"CabinetID": 2, "Capacity": 600},
    {"CabinetID": 3, "Capacity": 500},
    {"CabinetID": 4, "Capacity": 700},
    {"CabinetID": 5, "Capacity": 450},
    {"CabinetID": 6, "Capacity": 650},
    {"CabinetID": 7, "Capacity": 550},
    {"CabinetID": 8, "Capacity": 750},
    {"CabinetID": 9, "Capacity": 480},
    {"CabinetID": 10, "Capacity": 520}
  ],
  "products": [
    {"ProductName": "Espresso Beans", "Value": 100, "Weight": 1.0},
    {"ProductName": "Colombian Roast", "Value": 150, "Weight": 1.5},
    {"ProductName": "Arabica Blend", "Value": 80, "Weight": 1.2},
    {"ProductName": "French Roast", "Value": 120, "Weight": 1.3},
    {"ProductName": "Italian Roast", "Value": 130, "Weight": 1.4},
    {"ProductName": "House Blend", "Value": 110, "Weight": 1.1},
    {"ProductName": "Sumatra Coffee", "Value": 160, "Weight": 1.8},
    {"ProductName": "Mocha Java", "Value": 90, "Weight": 1.2},
    {"ProductName": "Hazelnut Flavor", "Value": 95, "Weight": 1.0},
    {"ProductName": "Caramel Blend", "Value": 105, "Weight": 1.3},
    {"ProductName": "Vanilla Flavor", "Value": 85, "Weight": 1.2},
    {"ProductName": "Cappuccino Mix", "Value": 140, "Weight": 1.5},
    {"ProductName": "Pumpkin Spice", "Value": 75, "Weight": 1.1},
    {"ProductName": "Decaf Roast", "Value": 60, "Weight": 1.0},
    {"ProductName": "Organic Roast", "Value": 170, "Weight": 1.6},
    {"ProductName": "Cold Brew", "Value": 115, "Weight": 1.4},
    {"ProductName": "Peruvian Blend", "Value": 155, "Weight": 1.7},
    {"ProductName": "Kenyan AA", "Value": 125, "Weight": 1.3}
  ]
}

##### Parameter Vectors and Matrices

- Cabinet capacities: $C = [400, 600, 500, 700, 450, 650, 550, 750, 480, 520]$
- Product values: $v = [100, 150, 80, 120, 130, 110, 160, 90, 95, 105, 85, 140, 75, 60, 170, 115, 155, 125]$
- Product weights: $w = [1.0, 1.5, 1.2, 1.3, 1.4, 1.1, 1.8, 1.2, 1.0, 1.3, 1.2, 1.5, 1.1, 1.0, 1.6, 1.4, 1.7, 1.3]$

##### Decision Variables

$x_{ij}$: Number of units of product $j$ placed in cabinet $i$, integer and non-negative.