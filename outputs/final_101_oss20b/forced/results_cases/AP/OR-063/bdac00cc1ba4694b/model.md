##### Objective Function: 

$\quad \quad \min \sum_{i=1}^{7} f_i y_i + \sum_{i=1}^{7} \sum_{j=1}^{7} c_{ij} x_{ij}$

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$\sum_{i=1}^{7} x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5,6,7\}$

###### 2. Activation Constraints: 

$x_{ij} \le d_j y_i \quad \forall i \in \{1,2,3,4,5,6,7\}, \; \forall j \in \{1,2,3,4,5,6,7\}$

###### 3. Non‑negativity Constraints: 

$x_{ij} \ge 0 \quad \forall i,j$

###### 4. Binary Constraints: 

$y_i \in \{0,1\} \quad \forall i$

###### 5. Variable Domain Constraints: 

$x_{ij} \in \mathbb{R} \quad \forall i,j$

###### 6. Fixed Cost Parameters: 

$f_i$ are given fixed costs for each warehouse.

###### 7. Transportation Cost Parameters: 

$c_{ij}$ are given per‑unit transportation costs from warehouse $i$ to customer $j$.

###### 8. Demand Parameters: 

$d_j$ are given demand quantities for each customer.

###### 9. Warehouse and Customer Sets: 

$W = \{S1,S2,S3,S4,S5,S6,S7\}$, $C = \{C1,C2,C3,C4,C5,C6,C7\}$

###### 10. Parameter Retrieval: 

$ f_i = \begin{cases}
102.33 & i=S1\\
94.92 & i=S2\\
91.83 & i=S3\\
98.71 & i=S4\\
95.73 & i=S5\\
99.96 & i=S6\\
98.16 & i=S7
\end{cases}$

$d_j = \begin{cases}
1083 & j=C1\\
776 & j=C2\\
16214 & j=C3\\
553 & j=C4\\
17106 & j=C5\\
594 & j=C6\\
732 & j=C7
\end{cases}$

$c_{ij} = \begin{cases}
1506.22 & (i=S1,j=C1)\\
70.90 & (i=S1,j=C2)\\
8.44 & (i=S1,j=C3)\\
260.27 & (i=S1,j=C4)\\
197.47 & (i=S1,j=C5)\\
71.71 & (i=S1,j=C6)\\
61.19 & (i=S1,j=C7)\\
1732.65 & (i=S2,j=C1)\\
1780.72 & (i=S2,j=C2)\\
567.44 & (i=S2,j=C3)\\
448.68 & (i=S2,j=C4)\\
29.00 & (i=S2,j=C5)\\
1484.91 & (i=S2,j=C6)\\
963.92 & (i=S2,j=C7)\\
115.66 & (i=S3,j=C1)\\
100.76 & (i=S3,j=C2)\\
64.68 & (i=S3,j=C3)\\
1324.53 & (i=S3,j=C4)\\
64.99 & (i=S3,j=C5)\\
134.88 & (i=S3,j=C6)\\
2102.83 & (i=S3,j=C7)\\
1254.78 & (i=S4,j=C1)\\
1115.63 & (i=S4,j=C2)\\
52.31 & (i=S4,j=C3)\\
1036.16 & (i=S4,j=C4)\\
892.63 & (i=S4,j=C5)\\
1464.04 & (i=S4,j=C6)\\
1383.41 & (i=S4,j=C7)\\
42.90 & (i=S5,j=C1)\\
891.01 & (i=S5,j=C2)\\
1013.94 & (i=S5,j=C3)\\
1128.72 & (i=S5,j=C4)\\
58.91 & (i=S5,j=C5)\\
42.89 & (i=S5,j=C6)\\
1570.31 & (i=S5,j=C7)\\
0.70 & (i=S6,j=C1)\\
139.46 & (i=S6,j=C2)\\
70.03 & (i=S6,j=C3)\\
79.15 & (i=S6,j=C4)\\
1482.00 & (i=S6,j=C5)\\
0.91 & (i=S6,j=C6)\\
110.46 & (i=S6,j=C7)\\
1732.30 & (i=S7,j=C1)\\
1780.44 & (i=S7,j=C2)\\
486.50 & (i=S7,j=C3)\\
523.74 & (i=S7,j=C4)\\
522.08 & (i=S7,j=C5)\\
82.48 & (i=S7,j=C6)\\
826.41 & (i=S7,j=C7)
\end{cases}$

###### 11. Retrieved Information

{
  "fixed_costs": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83,
    "S4": 98.71,
    "S5": 95.73,
    "S6": 99.96,
    "S7": 98.16
  },
  "demand": {
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
  },
  "warehouses": ["S1","S2","S3","S4","S5","S6","S7"],
  "customers": ["C1","C2","C3","C4","C5","C6","C7"]
}