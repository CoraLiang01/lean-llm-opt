##### Objective Function: 

$ \quad \quad \min \sum_{i} f_i \, y_i \;+\; \sum_{i}\sum_{j} c_{ij} \, x_{ij} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \quad \quad \sum_{i} x_{ij} \;=\; d_j , \quad \forall j $

###### 2. Supplier Activation Constraints: 

$ \quad \quad x_{ij} \;\leq\; d_j \, y_i , \quad \forall i,j $

###### 3. Non‑negativity and Integrality Constraints: 

$ \quad \quad x_{ij} \;\in\; \mathbb{Z}_{\geq 0} , \quad \forall i,j $  

$ \quad \quad y_i \;\in\; \{0,1\} , \quad \forall i $

##### Retrieved Information

```json
{
  "fixed_cost": [
    {"SupplierID":"S1","FixedCost":100.64},
    {"SupplierID":"S2","FixedCost":98.72},
    {"SupplierID":"S3","FixedCost":100.18},
    {"SupplierID":"S4","FixedCost":96.58},
    {"SupplierID":"S5","FixedCost":95.75},
    {"SupplierID":"S6","FixedCost":99.06},
    {"SupplierID":"S7","FixedCost":101.78},
    {"SupplierID":"S8","FixedCost":93.86}
  ],
  "transportation_costs": [
    {"SupplierID":"S1","DealershipID":"C1","CostPerVehicle":1091.04},
    {"SupplierID":"S1","DealershipID":"C2","CostPerVehicle":85.72},
    {"SupplierID":"S1","DealershipID":"C3","CostPerVehicle":99.08},
    {"SupplierID":"S1","DealershipID":"C4","CostPerVehicle":747.35},
    {"SupplierID":"S1","DealershipID":"C5","CostPerVehicle":893.86},
    {"SupplierID":"S1","DealershipID":"C6","CostPerVehicle":23.65},
    {"SupplierID":"S1","DealershipID":"C7","CostPerVehicle":15.11},
    {"SupplierID":"S1","DealershipID":"C8","CostPerVehicle":15.03},
    {"SupplierID":"S1","DealershipID":"C9","CostPerVehicle":497.88},
    {"SupplierID":"S2","DealershipID":"C1","CostPerVehicle":58.88},
    {"SupplierID":"S2","DealershipID":"C2","CostPerVehicle":1617.16},
    {"SupplierID":"S2","DealershipID":"C3","CostPerVehicle":1786.44},
    {"SupplierID":"S2","DealershipID":"C4","CostPerVehicle":951.81},
    {"SupplierID":"S2","DealershipID":"C5","CostPerVehicle":56.45},
    {"SupplierID":"S2","DealershipID":"C6","CostPerVehicle":642.77},
    {"SupplierID":"S2","DealershipID":"C7","CostPerVehicle":16.69},
    {"SupplierID":"S2","DealershipID":"C8","CostPerVehicle":0.63},
    {"SupplierID":"S2","DealershipID":"C9","CostPerVehicle":11.2},
    {"SupplierID":"S3","DealershipID":"C1","CostPerVehicle":110.47},
    {"SupplierID":"S3","DealershipID":"C2","CostPerVehicle":0.04},
    {"SupplierID":"S3","DealershipID":"C3","CostPerVehicle":38.89},
    {"SupplierID":"S3","DealershipID":"C4","CostPerVehicle":1397.95},
    {"SupplierID":"S3","DealershipID":"C5","CostPerVehicle":2361.45},
    {"SupplierID":"S3","DealershipID":"C6","CostPerVehicle":107.62},
    {"SupplierID":"S3","DealershipID":"C7","CostPerVehicle":1598.5},
    {"SupplierID":"S3","DealershipID":"C8","CostPerVehicle":76.41},
    {"SupplierID":"S3","DealershipID":"C9","CostPerVehicle":1382.84},
    {"SupplierID":"S4","DealershipID":"C1","CostPerVehicle":1458.85},
    {"SupplierID":"S4","DealershipID":"C2","CostPerVehicle":1049.27},
    {"SupplierID":"S4","DealershipID":"C3","CostPerVehicle":597.32},
    {"SupplierID":"S4","DealershipID":"C4","CostPerVehicle":1731.9},
    {"SupplierID":"S4","DealershipID":"C5","CostPerVehicle":69.09},
    {"SupplierID":"S4","DealershipID":"C6","CostPerVehicle":1227.17},
    {"SupplierID":"S4","DealershipID":"C7","CostPerVehicle":1187.55},
    {"SupplierID":"S4","DealershipID":"C8","CostPerVehicle":1017.16},
    {"SupplierID":"S4","DealershipID":"C9","CostPerVehicle":52.15},
    {"SupplierID":"S5","DealershipID":"C1","CostPerVehicle":0.38},
    {"SupplierID":"S5","DealershipID":"C2","CostPerVehicle":2315.52},
    {"SupplierID":"S5","DealershipID":"C3","CostPerVehicle":1313.06},
    {"SupplierID":"S5","DealershipID":"C4","CostPerVehicle":1253.71},
    {"SupplierID":"S5","DealershipID":"C5","CostPerVehicle":50.24},
    {"SupplierID":"S5","DealershipID":"C6","CostPerVehicle":29.19},
    {"SupplierID":"S5","DealershipID":"C7","CostPerVehicle":60.17},
    {"SupplierID":"S5","DealershipID":"C8","CostPerVehicle":1077.35},
    {"SupplierID":"S5","DealershipID":"C9","CostPerVehicle":70.11},
    {"SupplierID":"S6","DealershipID":"C1","CostPerVehicle":58.2},
    {"SupplierID":"S6","DealershipID":"C2","CostPerVehicle":1395.81},
    {"SupplierID":"S6","DealershipID":"C3","CostPerVehicle":84.6},
    {"SupplierID":"S6","DealershipID":"C4","CostPerVehicle":830.64},
    {"SupplierID":"S6","DealershipID":"C5","CostPerVehicle":1003.86},
    {"SupplierID":"S6","DealershipID":"C6","CostPerVehicle":631.17},
    {"SupplierID":"S6","DealershipID":"C7","CostPerVehicle":31.13},
    {"SupplierID":"S6","DealershipID":"C8","CostPerVehicle":1.4},
    {"SupplierID":"S6","DealershipID":"C9","CostPerVehicle":246.24},
    {"SupplierID":"S7","DealershipID":"C1","CostPerVehicle":1255.23},
    {"SupplierID":"S7","DealershipID":"C2","CostPerVehicle":1382.31},
    {"SupplierID":"S7","DealershipID":"C3","CostPerVehicle":78.79},
    {"SupplierID":"S7","DealershipID":"C4","CostPerVehicle":829.02},
    {"SupplierID":"S7","DealershipID":"C5","CostPerVehicle":67.31},
    {"SupplierID":"S7","DealershipID":"C6","CostPerVehicle":877.35},
    {"SupplierID":"S7","DealershipID":"C7","CostPerVehicle":185.28},
    {"SupplierID":"S7","DealershipID":"C8","CostPerVehicle":221.98},
    {"SupplierID":"S7","DealershipID":"C9","CostPerVehicle":0.05},
    {"SupplierID":"S8","DealershipID":"C1","CostPerVehicle":1990.09},
    {"SupplierID":"S8","DealershipID":"C2","CostPerVehicle":1.23},
    {"SupplierID":"S8","DealershipID":"C3","CostPerVehicle":38.97},
    {"SupplierID":"S8","DealershipID":"C4","CostPerVehicle":1396.35},
    {"SupplierID":"S8","DealershipID":"C5","CostPerVehicle":112.54},
    {"SupplierID":"S8","DealershipID":"C6","CostPerVehicle":107.54},
    {"SupplierID":"S8","DealershipID":"C7","CostPerVehicle":1596.74},
    {"SupplierID":"S8","DealershipID":"C8","CostPerVehicle":76.32},
    {"SupplierID":"S8","DealershipID":"C9","CostPerVehicle":1183.79}
  ],
  "demand": [
    {"DealershipID":"C1","Demand":4742532000},
    {"DealershipID":"C2","Demand":1600594000},
    {"DealershipID":"C3","Demand":5086889000},
    {"DealershipID":"C4","Demand":1027326000},
    {"DealershipID":"C5","Demand":11926044000},
    {"DealershipID":"C6","Demand":9058407000},
    {"DealershipID":"C7","Demand":5344367000},
    {"DealershipID":"C8","Demand":677201000},
    {"DealershipID":"C9","Demand":3236493000}
  ]
}
```