##### Objective Function: 

$\quad \quad \min \sum_{i} f_{i} \, y_{i} + \sum_{i} \sum_{j} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$\quad \quad \sum_{i} x_{ij} = d_{j}, \quad \forall j$

###### 2. Warehouse Activation Constraints: 

$\quad \quad x_{ij} \leq d_{j} \, y_{i}, \quad \forall i, j$

###### 3. Non-negativity and Integrality Constraints: 

$\quad \quad x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i, j$

$\quad \quad y_{i} \in \{0,1\}, \quad \forall i$

###### Retrieved Information

{
  "warehouses": [
    { "WarehouseID": "S1", "FixedCost": 102.33 },
    { "WarehouseID": "S2", "FixedCost": 94.92 },
    { "WarehouseID": "S3", "FixedCost": 91.83 }
  ],
  "musicians": [
    { "MusicianID": "C1", "Demand": 1083 },
    { "MusicianID": "C2", "Demand": 776 },
    { "MusicianID": "C3", "Demand": 16214 }
  ],
  "transportation_costs": [
    { "WarehouseID": "S1", "MusicianID": "C1", "CostPerUnit": 1506.22 },
    { "WarehouseID": "S1", "MusicianID": "C2", "CostPerUnit": 70.9 },
    { "WarehouseID": "S1", "MusicianID": "C3", "CostPerUnit": 8.44 },
    { "WarehouseID": "S2", "MusicianID": "C1", "CostPerUnit": 1732.65 },
    { "WarehouseID": "S2", "MusicianID": "C2", "CostPerUnit": 1780.72 },
    { "WarehouseID": "S2", "MusicianID": "C3", "CostPerUnit": 567.44 },
    { "WarehouseID": "S3", "MusicianID": "C1", "CostPerUnit": 115.66 },
    { "WarehouseID": "S3", "MusicianID": "C2", "CostPerUnit": 100.76 },
    { "WarehouseID": "S3", "MusicianID": "C3", "CostPerUnit": 64.68 }
  ]
}