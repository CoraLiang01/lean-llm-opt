##### Objective Function:

$\quad \min \left( \sum_{i=1}^7 f_i y_i + \sum_{i=1}^7 \sum_{j=1}^7 c_{ij} x_{ij} \right)$

where:
- $f_i$ is the fixed cost of opening warehouse $S_i$
- $y_i$ is a binary variable indicating if warehouse $S_i$ is open ($y_i \in \{0,1\}$)
- $c_{ij}$ is the transportation cost per unit from warehouse $S_i$ to customer $C_j$
- $x_{ij}$ is the quantity shipped from warehouse $S_i$ to customer $C_j$

##### Constraints

###### 1. Demand Satisfaction:

$\sum_{i=1}^7 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5,6,7\}$

where $d_j$ is the demand of customer $C_j$.

###### 2. Warehouse Activation:

$x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3,4,5,6,7\},\ \forall j \in \{1,2,3,4,5,6,7\}$

###### 3. Variable Domains:

$y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5,6,7\}$

$x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5,6,7\},\ \forall j \in \{1,2,3,4,5,6,7\}$

##### Retrieved Information

```json
{
  "warehouses": [
    "S1", "S2", "S3", "S4", "S5", "S6", "S7"
  ],
  "customers": [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7"
  ],
  "fixed_costs": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83,
    "S4": 98.71,
    "S5": 95.73,
    "S6": 99.96,
    "S7": 98.16
  },
  "demands": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214,
    "C4": 553,
    "C5": 17106,
    "C6": 594,
    "C7": 732
  },
  "transportation_costs": {
    "S1": {
      "C1": 1506.22,
      "C2": 70.90,
      "C3": 8.44,
      "C4": 260.27,
      "C5": 197.47,
      "C6": 71.71,
      "C7": 61.19
    },
    "S2": {
      "C1": 1732.65,
      "C2": 1780.72,
      "C3": 567.44,
      "C4": 448.68,
      "C5": 29.00,
      "C6": 1484.91,
      "C7": 963.92
    },
    "S3": {
      "C1": 115.66,
      "C2": 100.76,
      "C3": 64.68,
      "C4": 1324.53,
      "C5": 64.99,
      "C6": 134.88,
      "C7": 2102.83
    },
    "S4": {
      "C1": 1254.78,
      "C2": 1115.63,
      "C3": 52.31,
      "C4": 1036.16,
      "C5": 892.63,
      "C6": 1464.04,
      "C7": 1383.41
    },
    "S5": {
      "C1": 42.90,
      "C2": 891.01,
      "C3": 1013.94,
      "C4": 1128.72,
      "C5": 58.91,
      "C6": 42.89,
      "C7": 1570.31
    },
    "S6": {
      "C1": 0.70,
      "C2": 139.46,
      "C3": 70.03,
      "C4": 79.15,
      "C5": 1482.00,
      "C6": 0.91,
      "C7": 110.46
    },
    "S7": {
      "C1": 1732.30,
      "C2": 1780.44,
      "C3": 486.50,
      "C4": 523.74,
      "C5": 522.08,
      "C6": 82.48,
      "C7": 826.41
    }
  }
}
```

##### Variable Definitions

- $y_i$: Binary variable, $y_i = 1$ if warehouse $S_i$ is open, $0$ otherwise.
- $x_{ij}$: Quantity shipped from warehouse $S_i$ to customer $C_j$.

##### Sets

- Warehouses: $S = \{S1, S2, S3, S4, S5, S6, S7\}$
- Customers: $C = \{C1, C2, C3, C4, C5, C6, C7\}$

##### Parameters

- Fixed costs $f_i$ for each warehouse $S_i$ as above.
- Demands $d_j$ for each customer $C_j$ as above.
- Transportation costs $c_{ij}$ for each warehouse-customer pair as above.