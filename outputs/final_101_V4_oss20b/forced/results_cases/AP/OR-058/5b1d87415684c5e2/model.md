##### Objective Function: 

$ \quad \min \sum_{i=1}^{6} \sum_{j=1}^{6} c_{ij} x_{ij} + \sum_{i=1}^{6} f_{i} y_{i} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \sum_{i=1}^{6} x_{ij} \ge d_{j} \quad \forall j \in \{1,2,3,4,5,6\} $

###### 2. Supplier Activation Constraints: 

$ x_{ij} \le d_{j} \, y_{i} \quad \forall i \in \{1,2,3,4,5,6\}, \; \forall j \in \{1,2,3,4,5,6\} $

###### 3. Non‑negativity Constraints: 

$ x_{ij} \ge 0 \quad \forall i,j $

###### 4. Binary Constraints: 

$ y_{i} \in \{0,1\} \quad \forall i $

###### 5. Variable Domain Constraints: 

$ x_{ij} \in \mathbb{R} \quad \forall i,j $

##### Retrieved Information

{
  "fixed_costs": {
    "S1": 98.88,
    "S2": 99.73,
    "S3": 94.01,
    "S4": 93.77,
    "S5": 107.59,
    "S6": 112.65
  },
  "transportation_costs": {
    "S1": {
      "C1": 0.08,
      "C2": 52.33,
      "C3": 73.57,
      "C4": 1237.33,
      "C5": 0.07,
      "C6": 112.16
    },
    "S2": {
      "C1": 46.02,
      "C2": 175.23,
      "C3": 2026.83,
      "C4": 299.89,
      "C5": 966.53,
      "C6": 1590.42
    },
    "S3": {
      "C1": 1031.74,
      "C2": 78.13,
      "C3": 99.02,
      "C4": 277.07,
      "C5": 884.45,
      "C6": 1800.86
    },
    "S4": {
      "C1": 868.75,
      "C2": 94.2,
      "C3": 1776.34,
      "C4": 285.48,
      "C5": 868.85,
      "C6": 86.55
    },
    "S5": {
      "C1": 1577,
      "C2": 760.15,
      "C3": 2090.19,
      "C4": 43.2,
      "C5": 1577.12,
      "C6": 1095.17
    },
    "S6": {
      "C1": 49.14,
      "C2": 4.33,
      "C3": 2079.57,
      "C4": 277.04,
      "C5": 1032.01,
      "C6": 1543.49
    }
  },
  "demands": {
    "C1": 216,
    "C2": 216,
    "C3": 216,
    "C4": 144,
    "C5": 144,
    "C6": 144
  }
}