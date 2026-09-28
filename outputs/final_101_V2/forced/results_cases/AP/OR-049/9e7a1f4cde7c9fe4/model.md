##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the integer number of units of product $j$ placed on shelf $i$, and $v_j$ is the value of product $j$.

##### Constraints

###### 1. Shelf Capacity Constraints:

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

where $w_j$ is the weight of product $j$, and $C_i$ is the capacity of shelf $i$.

###### 2. Variable Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\}, \; j \in \{1,\ldots,20\}$

##### Retrieved Information

{
  "shelves": [
    {"ShelfID": "1", "Capacity": 5.0},
    {"ShelfID": "2", "Capacity": 7.0},
    {"ShelfID": "3", "Capacity": 6.0},
    {"ShelfID": "4", "Capacity": 8.0},
    {"ShelfID": "5", "Capacity": 5.5},
    {"ShelfID": "6", "Capacity": 9.0},
    {"ShelfID": "7", "Capacity": 6.5},
    {"ShelfID": "8", "Capacity": 7.5},
    {"ShelfID": "9", "Capacity": 8.2},
    {"ShelfID": "10", "Capacity": 5.7}
  ],
  "products": [
    {"ProductName": "Smartphone", "Value": 200, "Weight": 1.0},
    {"ProductName": "Laptop", "Value": 1500, "Weight": 5.0},
    {"ProductName": "Headphones", "Value": 100, "Weight": 0.5},
    {"ProductName": "Camera", "Value": 800, "Weight": 2.0},
    {"ProductName": "Smartwatch", "Value": 250, "Weight": 0.3},
    {"ProductName": "Tablet", "Value": 600, "Weight": 1.5},
    {"ProductName": "Bluetooth Speaker", "Value": 150, "Weight": 1.0},
    {"ProductName": "Keyboard", "Value": 80, "Weight": 0.8},
    {"ProductName": "Mouse", "Value": 50, "Weight": 0.2},
    {"ProductName": "Monitor", "Value": 300, "Weight": 3.0},
    {"ProductName": "Printer", "Value": 400, "Weight": 4.0},
    {"ProductName": "External Hard Drive", "Value": 120, "Weight": 0.5},
    {"ProductName": "Router", "Value": 60, "Weight": 0.3},
    {"ProductName": "Power Bank", "Value": 40, "Weight": 0.4},
    {"ProductName": "Memory Card", "Value": 30, "Weight": 0.05},
    {"ProductName": "USB Flash Drive", "Value": 25, "Weight": 0.02},
    {"ProductName": "Smart Home Hub", "Value": 100, "Weight": 0.6},
    {"ProductName": "Gaming Console", "Value": 500, "Weight": 4.0},
    {"ProductName": "Fitness Tracker", "Value": 90, "Weight": 0.2},
    {"ProductName": "E-Reader", "Value": 180, "Weight": 0.5}
  ]
}

##### Parameter Vectors and Matrices

- Shelves (indexed by $i$):  
  $C = [5.0,\, 7.0,\, 6.0,\, 8.0,\, 5.5,\, 9.0,\, 6.5,\, 7.5,\, 8.2,\, 5.7]$

- Products (indexed by $j$):  
  $v = [200,\, 1500,\, 100,\, 800,\, 250,\, 600,\, 150,\, 80,\, 50,\, 300,\, 400,\, 120,\, 60,\, 40,\, 30,\, 25,\, 100,\, 500,\, 90,\, 180]$  
  $w = [1.0,\, 5.0,\, 0.5,\, 2.0,\, 0.3,\, 1.5,\, 1.0,\, 0.8,\, 0.2,\, 3.0,\, 4.0,\, 0.5,\, 0.3,\, 0.4,\, 0.05,\, 0.02,\, 0.6,\, 4.0,\, 0.2,\, 0.5]$

- Decision variables:  
  $x_{ij}$: integer number of units of product $j$ on shelf $i$, for $i=1,\ldots,10$, $j=1,\ldots,20$