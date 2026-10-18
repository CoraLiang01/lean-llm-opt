##### Objective Function: 

$ \quad \quad \min \sum_{i=1}^{3} \sum_{j=1}^{3} c_{ij} x_{ij} + \sum_{i=1}^{3} f_{i} y_{i} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \sum_{i=1}^{3} x_{ij} = d_{j} \quad \forall j \in \{1,2,3\} $

###### 2. Warehouse Activation Constraints: 

$ x_{ij} \leq M_{j} y_{i} \quad \forall i \in \{1,2,3\}, \; \forall j \in \{1,2,3\} $

where $M_{j} = d_{j}$.

###### 3. Variable Constraints: 

$ x_{ij} \geq 0 \quad \forall i,j $  

$ y_{i} \in \{0,1\} \quad \forall i $

###### Retrieved Information

{
  "fixed_costs": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "transportation_costs": {
    "S1": {
      "C1": 1506.22,
      "C2": 70.9,
      "C3": 8.44
    },
    "S2": {
      "C1": 1732.65,
      "C2": 1780.72,
      "C3": 567.44
    },
    "S3": {
      "C1": 115.66,
      "C2": 100.76,
      "C3": 64.68
    }
  },
  "demand": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "warehouses": [
    "S1",
    "S2",
    "S3"
  ],
  "customers": [
    "C1",
    "C2",
    "C3"
  ]
}