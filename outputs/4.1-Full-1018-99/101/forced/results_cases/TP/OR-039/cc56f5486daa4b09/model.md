##### Sets

- $P$ = set of vehicle types (products):  
  $\{\text{Sedans},\ \text{SUVs},\ \text{Electric Vehicles},\ \text{Hybrid Vehicles},\ \text{Trucks},\ \text{Sports Cars},\ \text{Compact Cars},\ \text{Luxury Sedans},\ \text{Vans},\ \text{Pickup Trucks}\}$

- $W$ = set of warehouses:  
  $\{\text{Warehouse 1},\ \text{Warehouse 2},\ \text{Warehouse 3},\ \text{Warehouse 4},\ \text{Warehouse 5},\ \text{Warehouse 6},\ \text{Warehouse 7},\ \text{Warehouse 8},\ \text{Warehouse 9},\ \text{Warehouse 10}\}$

##### Parameters

- $v_p$ = value per unit of product $p$:

  | Product             | $v_p$ |
  |---------------------|-------|
  | Sedans              | 1200  |
  | SUVs                | 1800  |
  | Electric Vehicles   | 2500  |
  | Hybrid Vehicles     | 2000  |
  | Trucks              | 1500  |
  | Sports Cars         | 3000  |
  | Compact Cars        | 1000  |
  | Luxury Sedans       | 3500  |
  | Vans                | 1600  |
  | Pickup Trucks       | 1700  |

- $w_p$ = weight per unit of product $p$:

  | Product             | $w_p$ |
  |---------------------|-------|
  | Sedans              | 20    |
  | SUVs                | 15    |
  | Electric Vehicles   | 25    |
  | Hybrid Vehicles     | 18    |
  | Trucks              | 10    |
  | Sports Cars         | 5     |
  | Compact Cars        | 22    |
  | Luxury Sedans       | 8     |
  | Vans                | 12    |
  | Pickup Trucks       | 7     |

- $C_w$ = capacity of warehouse $w$:

  | Warehouse     | $C_w$ |
  |--------------|-------|
  | Warehouse 1  | 100   |
  | Warehouse 2  | 80    |
  | Warehouse 3  | 120   |
  | Warehouse 4  | 90    |
  | Warehouse 5  | 50    |
  | Warehouse 6  | 30    |
  | Warehouse 7  | 110   |
  | Warehouse 8  | 40    |
  | Warehouse 9  | 60    |
  | Warehouse 10 | 35    |

##### Decision Variables

- $x_{pw} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ stored in warehouse $w$ (integer, $\geq 0$).

##### Objective

$\max \sum_{p \in P} \sum_{w \in W} v_p \, x_{pw}$

##### Constraints

1. **Warehouse capacity constraints:**  
   For each warehouse $w \in W$,
   $$
   \sum_{p \in P} w_p \, x_{pw} \leq C_w
   $$

2. **Non-negativity and integrality:**  
   $$
   x_{pw} \in \mathbb{Z}_{\geq 0} \qquad \forall p \in P,\, w \in W
   $$

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