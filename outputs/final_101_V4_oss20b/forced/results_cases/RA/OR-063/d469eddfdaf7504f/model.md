##### Objective Function: 

$ \quad \quad \min \sum_{i} f_{i} \, y_{i} + \sum_{i} \sum_{j} c_{ij} \, x_{ij} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \quad \quad \sum_{i} x_{ij} = d_{j}, \quad \forall j $

###### 2. Warehouse Activation Constraints: 

$ \quad \quad x_{ij} \leq d_{j} \, y_{i}, \quad \forall i, j $

###### 3. Non-negativity and Integrality Constraints: 

$ \quad \quad x_{ij} \geq 0, \quad \forall i, j $  

$ \quad \quad y_{i} \in \{0,1\}, \quad \forall i $

###### Retrieved Information

{
  "fixed_cost": [
    { "WarehouseID": "S1", "FixedCost": 102.33 },
    { "WarehouseID": "S2", "FixedCost": 94.92 },
    { "WarehouseID": "S3", "FixedCost": 91.83 },
    { "WarehouseID": "S4", "FixedCost": 98.70999999999999 },
    { "WarehouseID": "S5", "FixedCost": 95.73 },
    { "WarehouseID": "S6", "FixedCost": 99.95999999999999 },
    { "WarehouseID": "S7", "FixedCost": 98.16 }
  ],
  "transportation_costs": [
    { "WarehouseID": "S1", "CustomerID": "C1", "CostPerUnit": 1506.22 },
    { "WarehouseID": "S1", "CustomerID": "C2", "CostPerUnit": 70.9 },
    { "WarehouseID": "S1", "CustomerID": "C3", "CostPerUnit": 8.44 },
    { "WarehouseID": "S1", "CustomerID": "C4", "CostPerUnit": 260.27 },
    { "WarehouseID": "S1", "CustomerID": "C5", "CostPerUnit": 197.47 },
    { "WarehouseID": "S1", "CustomerID": "C6", "CostPerUnit": 71.71 },
    { "WarehouseID": "S1", "CustomerID": "C7", "CostPerUnit": 61.19 },
    { "WarehouseID": "S2", "CustomerID": "C1", "CostPerUnit": 1732.65 },
    { "WarehouseID": "S2", "CustomerID": "C2", "CostPerUnit": 1780.72 },
    { "WarehouseID": "S2", "CustomerID": "C3", "CostPerUnit": 567.44 },
    { "WarehouseID": "S2", "CustomerID": "C4", "CostPerUnit": 448.68 },
    { "WarehouseID": "S2", "CustomerID": "C5", "CostPerUnit": 29 },
    { "WarehouseID": "S2", "CustomerID": "C6", "CostPerUnit": 1484.91 },
    { "WarehouseID": "S2", "CustomerID": "C7", "CostPerUnit": 963.92 },
    { "WarehouseID": "S3", "CustomerID": "C1", "CostPerUnit": 115.66 },
    { "WarehouseID": "S3", "CustomerID": "C2", "CostPerUnit": 100.76 },
    { "WarehouseID": "S3", "CustomerID": "C3", "CostPerUnit": 64.68 },
    { "WarehouseID": "S3", "CustomerID": "C4", "CostPerUnit": 1324.53 },
    { "WarehouseID": "S3", "CustomerID": "C5", "CostPerUnit": 64.99 },
    { "WarehouseID": "S3", "CustomerID": "C6", "CostPerUnit": 134.88 },
    { "WarehouseID": "S3", "CustomerID": "C7", "CostPerUnit": 2102.83 },
    { "WarehouseID": "S4", "CustomerID": "C1", "CostPerUnit": 1254.78 },
    { "WarehouseID": "S4", "CustomerID": "C2", "CostPerUnit": 1115.63 },
    { "WarehouseID": "S4", "CustomerID": "C3", "CostPerUnit": 52.31 },
    { "WarehouseID": "S4", "CustomerID": "C4", "CostPerUnit": 1036.16 },
    { "WarehouseID": "S4", "CustomerID": "C5", "CostPerUnit": 892.63 },
    { "WarehouseID": "S4", "CustomerID": "C6", "CostPerUnit": 1464.04 },
    { "WarehouseID": "S4", "CustomerID": "C7", "CostPerUnit": 1383.41 },
    { "WarehouseID": "S5", "CustomerID": "C1", "CostPerUnit": 42.9 },
    { "WarehouseID": "S5", "CustomerID": "C2", "CostPerUnit": 891.01 },
    { "WarehouseID": "S5", "CustomerID": "C3", "CostPerUnit": 1013.94 },
    { "WarehouseID": "S5", "CustomerID": "C4", "CostPerUnit": 1128.72 },
    { "WarehouseID": "S5", "CustomerID": "C5", "CostPerUnit": 58.91 },
    { "WarehouseID": "S5", "CustomerID": "C6", "CostPerUnit": 42.89 },
    { "WarehouseID": "S5", "CustomerID": "C7", "CostPerUnit": 1570.31 },
    { "WarehouseID": "S6", "CustomerID": "C1", "CostPerUnit": 0.7 },
    { "WarehouseID": "S6", "CustomerID": "C2", "CostPerUnit": 139.46 },
    { "WarehouseID": "S6", "CustomerID": "C3", "CostPerUnit": 70.03 },
    { "WarehouseID": "S6", "CustomerID": "C4", "CostPerUnit": 79.15 },
    { "WarehouseID": "S6", "CustomerID": "C5", "CostPerUnit": 1482 },
    { "WarehouseID": "S6", "CustomerID": "C6", "CostPerUnit": 0.91 },
    { "WarehouseID": "S6", "CustomerID": "C7", "CostPerUnit": 110.46 },
    { "WarehouseID": "S7", "CustomerID": "C1", "CostPerUnit": 1732.3 },
    { "WarehouseID": "S7", "CustomerID": "C2", "CostPerUnit": 1780.44 },
    { "WarehouseID": "S7", "CustomerID": "C3", "CostPerUnit": 486.5 },
    { "WarehouseID": "S7", "CustomerID": "C4", "CostPerUnit": 523.74 },
    { "WarehouseID": "S7", "CustomerID": "C5", "CostPerUnit": 522.08 },
    { "WarehouseID": "S7", "CustomerID": "C6", "CostPerUnit": 82.48 },
    { "WarehouseID": "S7", "CustomerID": "C7", "CostPerUnit": 826.41 }
  ],
  "demands": [
    { "CustomerID": "C1", "Demand": 1083 },
    { "CustomerID": "C2", "Demand": 776 },
    { "CustomerID": "C3", "Demand": 16214 },
    { "CustomerID": "C4", "Demand": 553 },
    { "CustomerID": "C5", "Demand": 17106 },
    { "CustomerID": "C6", "Demand": 594 },
    { "CustomerID": "C7", "Demand": 732 }
  ]
}