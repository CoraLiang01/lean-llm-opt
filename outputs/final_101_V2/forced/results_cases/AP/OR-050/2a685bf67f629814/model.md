##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

##### Constraints

###### 1. Display Capacity Constraints:

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

###### 2. Minimum Quantity of First Product (Smartphone):

$\sum_{i=1}^{10} x_{i1} \geq 5$

###### 3. Variable Constraints:

$x_{ij} \geq 0$ and integer, $\quad \forall i \in \{1,\ldots,10\}, \; j \in \{1,\ldots,20\}$

##### Retrieved Information

{
  "displays": [
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

Where:
- $x_{ij}$: Number of units of product $j$ placed on display $i$
- $v_j$: Value of product $j$ (see products list above)
- $w_j$: Weight of product $j$ (see products list above)
- $C_i$: Capacity of display $i$ (see displays list above)
- The first product (Smartphone) corresponds to $j=1$