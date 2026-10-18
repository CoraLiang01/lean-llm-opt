##### Sets

Let  
$F = \{S1, S2, S3, S4, S5, S6, S7, S8\}$: set of suppliers  
$C = \{C1, C2, C3, C4, C5, C6, C7, C8, C9\}$: set of dealerships

##### Parameters

- $d_j$: demand of dealership $j \in C$
  - $d_{C1} = 4,\!742,\!532,\!000$
  - $d_{C2} = 1,\!600,\!594,\!000$
  - $d_{C3} = 5,\!086,\!889,\!000$
  - $d_{C4} = 1,\!027,\!326,\!000$
  - $d_{C5} = 11,\!926,\!044,\!000$
  - $d_{C6} = 9,\!058,\!407,\!000$
  - $d_{C7} = 5,\!344,\!367,\!000$
  - $d_{C8} = 677,\!201,\!000$
  - $d_{C9} = 3,\!236,\!493,\!000$

- $f_i$: fixed cost to open supplier $i \in F$
  - $f_{S1} = 100.64$
  - $f_{S2} = 98.72$
  - $f_{S3} = 100.18$
  - $f_{S4} = 96.58$
  - $f_{S5} = 95.75$
  - $f_{S6} = 99.06$
  - $f_{S7} = 101.78$
  - $f_{S8} = 93.86$

- $c_{ij}$: transportation cost per vehicle from supplier $i$ to dealership $j$

|        |  C1     |   C2    |   C3    |   C4    |   C5    |   C6    |   C7    |   C8    |   C9    |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| S1     | 1091.04 | 85.72   | 99.08   | 747.35  | 893.86  | 23.65   | 15.11   | 15.03   | 497.88  |
| S2     | 58.88   | 1617.16 | 1786.44 | 951.81  | 56.45   | 642.77  | 16.69   | 0.63    | 11.2    |
| S3     | 110.47  | 0.04    | 38.89   | 1397.95 | 2361.45 | 107.62  | 1598.5  | 76.41   | 1382.84 |
| S4     | 1458.85 | 1049.27 | 597.32  | 1731.9  | 69.09   | 1227.17 | 1187.55 | 1017.16 | 52.15   |
| S5     | 0.38    | 2315.52 | 1313.06 | 1253.71 | 50.24   | 29.19   | 60.17   | 1077.35 | 70.11   |
| S6     | 58.2    | 1395.81 | 84.6    | 830.64  | 1003.86 | 631.17  | 31.13   | 1.4     | 246.24  |
| S7     | 1255.23 | 1382.31 | 78.79   | 829.02  | 67.31   | 877.35  | 185.28  | 221.98  | 0.05    |
| S8     | 1990.09 | 1.23    | 38.97   | 1396.35 | 112.54  | 107.54  | 1596.74 | 76.32   | 1183.79 |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise, for all $i \in F$
- $x_{ij} \geq 0$: number of vehicles supplied from supplier $i$ to dealership $j$, for all $i \in F$, $j \in C$

##### Objective

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each dealership $j \in C$,
   $$
   \sum_{i \in F} x_{ij} = d_j
   $$

2. **Supplier activation:**  
   For all $i \in F$, $j \in C$,
   $$
   x_{ij} \leq d_j y_i
   $$
   (A supplier can only supply to a dealership if it is open; $d_j$ is an upper bound.)

3. **Variable domains:**  
   $$
   y_i \in \{0,1\} \quad \forall i \in F
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in F,\, j \in C
   $$

##### Complete Numerical Formulation

Let $F = \{S1, S2, S3, S4, S5, S6, S7, S8\}$, $C = \{C1, C2, C3, C4, C5, C6, C7, C8, C9\}$.

Minimize
$$
100.64\,y_{S1} + 98.72\,y_{S2} + 100.18\,y_{S3} + 96.58\,y_{S4} + 95.75\,y_{S5} + 99.06\,y_{S6} + 101.78\,y_{S7} + 93.86\,y_{S8}
$$
$$
+ \sum_{i \in F} \sum_{j \in C} c_{ij} x_{ij}
$$
where $c_{ij}$ is as in the table above.

Subject to, for each $j \in C$:
$$
x_{S1,j} + x_{S2,j} + x_{S3,j} + x_{S4,j} + x_{S5,j} + x_{S6,j} + x_{S7,j} + x_{S8,j} = d_j
$$

For all $i \in F$, $j \in C$:
$$
x_{ij} \leq d_j y_i
$$

$y_i \in \{0,1\}$ for all $i \in F$  
$x_{ij} \geq 0$ for all $i \in F$, $j \in C$

###### Retrieved Information

{
  "customers": [
    {"customer": "C1", "demand": "4742532000"},
    {"customer": "C2", "demand": "1600594000"},
    {"customer": "C3", "demand": "5086889000"},
    {"customer": "C4", "demand": "1027326000"},
    {"customer": "C5", "demand": "11926044000"},
    {"customer": "C6", "demand": "9058407000"},
    {"customer": "C7", "demand": "5344367000"},
    {"customer": "C8", "demand": "677201000"},
    {"customer": "C9", "demand": "3236493000"}
  ],
  "fixed_costs": [
    {"supplier": "S1", "fixed_costs": "100.64"},
    {"supplier": "S2", "fixed_costs": "98.72"},
    {"supplier": "S3", "fixed_costs": "100.18"},
    {"supplier": "S4", "fixed_costs": "96.58"},
    {"supplier": "S5", "fixed_costs": "95.75"},
    {"supplier": "S6", "fixed_costs": "99.06"},
    {"supplier": "S7", "fixed_costs": "101.78"},
    {"supplier": "S8", "fixed_costs": "93.86"}
  ],
  "transportation_costs": [
    {"supplier": "S1", "C1": "1091.04", "C2": "85.72", "C3": "99.08", "C4": "747.35", "C5": "893.86", "C6": "23.65", "C7": "15.11", "C8": "15.03", "C9": "497.88"},
    {"supplier": "S2", "C1": "58.88", "C2": "1617.16", "C3": "1786.44", "C4": "951.81", "C5": "56.45", "C6": "642.77", "C7": "16.69", "C8": "0.63", "C9": "11.2"},
    {"supplier": "S3", "C1": "110.47", "C2": "0.04", "C3": "38.89", "C4": "1397.95", "C5": "2361.45", "C6": "107.62", "C7": "1598.5", "C8": "76.41", "C9": "1382.84"},
    {"supplier": "S4", "C1": "1458.85", "C2": "1049.27", "C3": "597.32", "C4": "1731.9", "C5": "69.09", "C6": "1227.17", "C7": "1187.55", "C8": "1017.16", "C9": "52.15"},
    {"supplier": "S5", "C1": "0.38", "C2": "2315.52", "C3": "1313.06", "C4": "1253.71", "C5": "50.24", "C6": "29.19", "C7": "60.17", "C8": "1077.35", "C9": "70.11"},
    {"supplier": "S6", "C1": "58.2", "C2": "1395.81", "C3": "84.6", "C4": "830.64", "C5": "1003.86", "C6": "631.17", "C7": "31.13", "C8": "1.4", "C9": "246.24"},
    {"supplier": "S7", "C1": "1255.23", "C2": "1382.31", "C3": "78.79", "C4": "829.02", "C5": "67.31", "C6": "877.35", "C7": "185.28", "C8": "221.98", "C9": "0.05"},
    {"supplier": "S8", "C1": "1990.09", "C2": "1.23", "C3": "38.97", "C4": "1396.35", "C5": "112.54", "C6": "107.54", "C7": "1596.74", "C8": "76.32", "C9": "1183.79"}
  ]
}