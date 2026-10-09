##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the number of units of product $j$ placed on display $i$, and $v_j$ is the value of product $j$.

##### Constraints:

###### 1. Display Capacity Constraints:

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

where $w_j$ is the weight of product $j$, and $C_i$ is the capacity of display $i$.

###### 2. Minimum Quantity of First Product (Smartphone):

$\sum_{i=1}^{10} x_{i,1} \geq 5$

###### 3. Non-negativity and Integrality:

$x_{ij} \geq 0$ and integer, $\quad \forall i \in \{1,\ldots,10\}, \; j \in \{1,\ldots,20\}$

---

##### Retrieved Information

```json
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
```

- Displays (with capacities):  
  1: 5.0, 2: 7.0, 3: 6.0, 4: 8.0, 5: 5.5, 6: 9.0, 7: 6.5, 8: 7.5, 9: 8.2, 10: 5.7

- Products (with values and weights, in order):  
  1. Smartphone: value 200, weight 1.0  
  2. Laptop: value 1500, weight 5.0  
  3. Headphones: value 100, weight 0.5  
  4. Camera: value 800, weight 2.0  
  5. Smartwatch: value 250, weight 0.3  
  6. Tablet: value 600, weight 1.5  
  7. Bluetooth Speaker: value 150, weight 1.0  
  8. Keyboard: value 80, weight 0.8  
  9. Mouse: value 50, weight 0.2  
  10. Monitor: value 300, weight 3.0  
  11. Printer: value 400, weight 4.0  
  12. External Hard Drive: value 120, weight 0.5  
  13. Router: value 60, weight 0.3  
  14. Power Bank: value 40, weight 0.4  
  15. Memory Card: value 30, weight 0.05  
  16. USB Flash Drive: value 25, weight 0.02  
  17. Smart Home Hub: value 100, weight 0.6  
  18. Gaming Console: value 500, weight 4.0  
  19. Fitness Tracker: value 90, weight 0.2  
  20. E-Reader: value 180, weight 0.5

- Decision variables: $x_{ij}$ = number of units of product $j$ placed on display $i$, for $i=1,\ldots,10$, $j=1,\ldots,20$.