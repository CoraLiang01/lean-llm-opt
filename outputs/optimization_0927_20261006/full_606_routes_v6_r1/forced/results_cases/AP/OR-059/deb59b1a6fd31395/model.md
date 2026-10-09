##### Objective Function:

$\quad \min \sum_{i=1}^8 f_i y_i + \sum_{i=1}^8 \sum_{j=1}^9 c_{ij} x_{ij}$

##### Constraints

###### 1. Demand Satisfaction (for each dealership $j$):

$\sum_{i=1}^8 x_{ij} = d_j \quad \forall j \in \{1,2,\ldots,9\}$

###### 2. Supply Activation (vehicles can only be supplied from open suppliers):

$x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,\ldots,8\},\ \forall j \in \{1,2,\ldots,9\}$

###### 3. Variable Domains:

$y_i \in \{0,1\} \quad \forall i \in \{1,2,\ldots,8\}$

$x_{ij} \geq 0 \quad \forall i \in \{1,2,\ldots,8\},\ \forall j \in \{1,2,\ldots,9\}$

---

##### Retrieved Information

```json
{
  "suppliers": [
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"
  ],
  "dealerships": [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"
  ],
  "fixed_costs": {
    "S1": 100.64,
    "S2": 98.72,
    "S3": 100.18,
    "S4": 96.58,
    "S5": 95.75,
    "S6": 99.06,
    "S7": 101.78,
    "S8": 93.86
  },
  "demands": {
    "C1": 4742532000,
    "C2": 1600594000,
    "C3": 5086889000,
    "C4": 1027326000,
    "C5": 11926044000,
    "C6": 9058407000,
    "C7": 5344367000,
    "C8": 677201000,
    "C9": 3236493000
  },
  "transportation_costs": {
    "S1": {
      "C1": 1091.04, "C2": 85.72, "C3": 99.08, "C4": 747.35, "C5": 893.86, "C6": 23.65, "C7": 15.11, "C8": 15.03, "C9": 497.88
    },
    "S2": {
      "C1": 58.88, "C2": 1617.16, "C3": 1786.44, "C4": 951.81, "C5": 56.45, "C6": 642.77, "C7": 16.69, "C8": 0.63, "C9": 11.2
    },
    "S3": {
      "C1": 110.47, "C2": 0.04, "C3": 38.89, "C4": 1397.95, "C5": 2361.45, "C6": 107.62, "C7": 1598.5, "C8": 76.41, "C9": 1382.84
    },
    "S4": {
      "C1": 1458.85, "C2": 1049.27, "C3": 597.32, "C4": 1731.9, "C5": 69.09, "C6": 1227.17, "C7": 1187.55, "C8": 1017.16, "C9": 52.15
    },
    "S5": {
      "C1": 0.38, "C2": 2315.52, "C3": 1313.06, "C4": 1253.71, "C5": 50.24, "C6": 29.19, "C7": 60.17, "C8": 1077.35, "C9": 70.11
    },
    "S6": {
      "C1": 58.2, "C2": 1395.81, "C3": 84.6, "C4": 830.64, "C5": 1003.86, "C6": 631.17, "C7": 31.13, "C8": 1.4, "C9": 246.24
    },
    "S7": {
      "C1": 1255.23, "C2": 1382.31, "C3": 78.79, "C4": 829.02, "C5": 67.31, "C6": 877.35, "C7": 185.28, "C8": 221.98, "C9": 0.05
    },
    "S8": {
      "C1": 1990.09, "C2": 1.23, "C3": 38.97, "C4": 1396.35, "C5": 112.54, "C6": 107.54, "C7": 1596.74, "C8": 76.32, "C9": 1183.79
    }
  }
}
```

- $f_i$ is the fixed cost for opening supplier $i$ (see "fixed_costs" above).
- $c_{ij}$ is the transportation cost per vehicle from supplier $i$ to dealership $j$ (see "transportation_costs" above).
- $d_j$ is the demand for dealership $j$ (see "demands" above).
- $y_i$ is a binary variable indicating if supplier $i$ is open.
- $x_{ij}$ is the number of vehicles supplied from supplier $i$ to dealership $j$.

All parameters and indices are as retrieved from the CSV files.