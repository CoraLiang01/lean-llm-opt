##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the number of units of product $j$ placed on shelf $i$, and $v_j$ is the value of product $j$.

##### Constraints:

###### 1. Shelf Capacity Constraints:

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

where $w_j$ is the weight of product $j$, and $C_i$ is the capacity of shelf $i$.

###### 2. Variable Domain Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}, \forall j \in \{1,2,\ldots,20\}$

##### Retrieved Information

{
  "shelves": [
    {"ShelfID": 1, "Capacity": 5.0},
    {"ShelfID": 2, "Capacity": 7.0},
    {"ShelfID": 3, "Capacity": 6.0},
    {"ShelfID": 4, "Capacity": 8.0},
    {"ShelfID": 5, "Capacity": 5.5},
    {"ShelfID": 6, "Capacity": 9.0},
    {"ShelfID": 7, "Capacity": 6.5},
    {"ShelfID": 8, "Capacity": 7.5},
    {"ShelfID": 9, "Capacity": 8.2},
    {"ShelfID": 10, "Capacity": 5.7}
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

##### Parameter Definitions

Let $i \in \{1,2,\ldots,10\}$ index shelves, and $j \in \{1,2,\ldots,20\}$ index products, with the following mappings:

- Shelves:
  1. Shelf 1: $C_1 = 5.0$
  2. Shelf 2: $C_2 = 7.0$
  3. Shelf 3: $C_3 = 6.0$
  4. Shelf 4: $C_4 = 8.0$
  5. Shelf 5: $C_5 = 5.5$
  6. Shelf 6: $C_6 = 9.0$
  7. Shelf 7: $C_7 = 6.5$
  8. Shelf 8: $C_8 = 7.5$
  9. Shelf 9: $C_9 = 8.2$
  10. Shelf 10: $C_{10} = 5.7$

- Products (with $v_j$ and $w_j$):

  1. Smartphone: $v_1 = 200$, $w_1 = 1.0$
  2. Laptop: $v_2 = 1500$, $w_2 = 5.0$
  3. Headphones: $v_3 = 100$, $w_3 = 0.5$
  4. Camera: $v_4 = 800$, $w_4 = 2.0$
  5. Smartwatch: $v_5 = 250$, $w_5 = 0.3$
  6. Tablet: $v_6 = 600$, $w_6 = 1.5$
  7. Bluetooth Speaker: $v_7 = 150$, $w_7 = 1.0$
  8. Keyboard: $v_8 = 80$, $w_8 = 0.8$
  9. Mouse: $v_9 = 50$, $w_9 = 0.2$
  10. Monitor: $v_{10} = 300$, $w_{10} = 3.0$
  11. Printer: $v_{11} = 400$, $w_{11} = 4.0$
  12. External Hard Drive: $v_{12} = 120$, $w_{12} = 0.5$
  13. Router: $v_{13} = 60$, $w_{13} = 0.3$
  14. Power Bank: $v_{14} = 40$, $w_{14} = 0.4$
  15. Memory Card: $v_{15} = 30$, $w_{15} = 0.05$
  16. USB Flash Drive: $v_{16} = 25$, $w_{16} = 0.02$
  17. Smart Home Hub: $v_{17} = 100$, $w_{17} = 0.6$
  18. Gaming Console: $v_{18} = 500$, $w_{18} = 4.0$
  19. Fitness Tracker: $v_{19} = 90$, $w_{19} = 0.2$
  20. E-Reader: $v_{20} = 180$, $w_{20} = 0.5$

##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$.

##### Complete Model

$\boxed{
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\end{align*}
}$