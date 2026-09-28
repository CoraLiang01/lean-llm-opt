##### Objective Function:

$\quad \max \sum_{w=1}^{10} \sum_{i=1}^{10} v_i \, x_{iw}$

where:
- $x_{iw}$ = number of vehicles of type $i$ stored in warehouse $w$ (integer decision variable)
- $v_i$ = value (benefit coefficient) of vehicle type $i$

##### Constraints

###### 1. Warehouse Capacity Constraints:

$\sum_{i=1}^{10} w_i \, x_{iw} \leq C_w \quad \forall w \in \{1,2,\ldots,10\}$

where:
- $w_i$ = weight (space requirement) of vehicle type $i$
- $C_w$ = capacity of warehouse $w$

###### 2. Variable Constraints:

$x_{iw} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\}, \forall w \in \{1,\ldots,10\}$

##### Retrieved Information

{
  "warehouses": [
    {"Warehouse ID": "Warehouse 1", "Capacity": 100},
    {"Warehouse ID": "Warehouse 2", "Capacity": 80},
    {"Warehouse ID": "Warehouse 3", "Capacity": 120},
    {"Warehouse ID": "Warehouse 4", "Capacity": 90},
    {"Warehouse ID": "Warehouse 5", "Capacity": 50},
    {"Warehouse ID": "Warehouse 6", "Capacity": 30},
    {"Warehouse ID": "Warehouse 7", "Capacity": 110},
    {"Warehouse ID": "Warehouse 8", "Capacity": 40},
    {"Warehouse ID": "Warehouse 9", "Capacity": 60},
    {"Warehouse ID": "Warehouse 10", "Capacity": 35}
  ],
  "products": [
    {"ProductName": "Sedans", "Value": 1200, "Weight": 20},
    {"ProductName": "SUVs", "Value": 1800, "Weight": 15},
    {"ProductName": "Electric Vehicles", "Value": 2500, "Weight": 25},
    {"ProductName": "Hybrid Vehicles", "Value": 2000, "Weight": 18},
    {"ProductName": "Trucks", "Value": 1500, "Weight": 10},
    {"ProductName": "Sports Cars", "Value": 3000, "Weight": 5},
    {"ProductName": "Compact Cars", "Value": 1000, "Weight": 22},
    {"ProductName": "Luxury Sedans", "Value": 3500, "Weight": 8},
    {"ProductName": "Vans", "Value": 1600, "Weight": 12},
    {"ProductName": "Pickup Trucks", "Value": 1700, "Weight": 7}
  ]
}

##### Parameter Vectors and Matrices

- Vehicle types $i = 1,\ldots,10$:
    1. Sedans: $v_1 = 1200$, $w_1 = 20$
    2. SUVs: $v_2 = 1800$, $w_2 = 15$
    3. Electric Vehicles: $v_3 = 2500$, $w_3 = 25$
    4. Hybrid Vehicles: $v_4 = 2000$, $w_4 = 18$
    5. Trucks: $v_5 = 1500$, $w_5 = 10$
    6. Sports Cars: $v_6 = 3000$, $w_6 = 5$
    7. Compact Cars: $v_7 = 1000$, $w_7 = 22$
    8. Luxury Sedans: $v_8 = 3500$, $w_8 = 8$
    9. Vans: $v_9 = 1600$, $w_9 = 12$
    10. Pickup Trucks: $v_{10} = 1700$, $w_{10} = 7$

- Warehouses $w = 1,\ldots,10$:
    1. Warehouse 1: $C_1 = 100$
    2. Warehouse 2: $C_2 = 80$
    3. Warehouse 3: $C_3 = 120$
    4. Warehouse 4: $C_4 = 90$
    5. Warehouse 5: $C_5 = 50$
    6. Warehouse 6: $C_6 = 30$
    7. Warehouse 7: $C_7 = 110$
    8. Warehouse 8: $C_8 = 40$
    9. Warehouse 9: $C_9 = 60$
    10. Warehouse 10: $C_{10} = 35$