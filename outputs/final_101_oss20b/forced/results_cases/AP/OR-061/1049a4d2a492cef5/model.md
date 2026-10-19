##### Objective Function: 

$ \quad \quad \min \sum_{i=1}^{5} \sum_{j=1}^{5} c_{ij} x_{ij} + \sum_{i=1}^{5} f_{i} y_{i} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \sum_{i=1}^{5} x_{ij} = d_{j} \quad \forall j \in \{1,2,3,4,5\} $

###### 2. Supplier Activation Linking Constraints: 

$ x_{ij} \le d_{j} \, y_{i} \quad \forall i \in \{1,2,3,4,5\}, \; \forall j \in \{1,2,3,4,5\} $

###### 3. Non‑negativity Constraints: 

$ x_{ij} \ge 0 \quad \forall i \in \{1,2,3,4,5\}, \; \forall j \in \{1,2,3,4,5\} $

###### 4. Binary Constraints: 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,3,4,5\} $

###### 5. Variable Domain Constraints: 

$ x_{ij} \in \mathbb{R} \quad \forall i \in \{1,2,3,4,5\}, \; \forall j \in \{1,2,3,4,5\} $

###### 6. Fixed Cost Parameters: 

$ f_{i} = \begin{cases}
97.65 & i=1 \\
99.76 & i=2 \\
100.76 & i=3 \\
105.32 & i=4 \\
98.88 & i=5
\end{cases} $

###### 7. Transportation Cost Parameters: 

$ c_{ij} = \begin{cases}
150.74 & (i=1,j=1) \\
0.02 & (i=1,j=2) \\
49.13 & (i=1,j=3) \\
2080.15 & (i=1,j=4) \\
426.40 & (i=1,j=5) \\
233.05 & (i=2,j=1) \\
97.73 & (i=2,j=2) \\
49.84 & (i=2,j=3) \\
1982.39 & (i=2,j=4) \\
23.96 & (i=2,j=5) \\
55.68 & (i=3,j=1) \\
935.61 & (i=3,j=2) \\
4.03 & (i=3,j=3) \\
73.09 & (i=3,j=4) \\
525.32 & (i=3,j=5) \\
1483.82 & (i=4,j=1) \\
1801.08 & (i=4,j=2) \\
112.16 & (i=4,j=3) \\
816.05 & (i=4,j=4) \\
107.01 & (i=4,j=5) \\
1119.47 & (i=5,j=1) \\
884.31 & (i=5,j=2) \\
0.08 & (i=5,j=3) \\
1544.95 & (i=5,j=4) \\
543.67 & (i=5,j=5)
\end{cases} $

###### 8. Demand Parameters: 

$ d_{j} = \begin{cases}
143 & j=1 \\
6 & j=2 \\
10 & j=3 \\
25 & j=4 \\
3 & j=5
\end{cases} $

###### Retrieved Information

{
  "fixed_costs": {
    "S1": 97.65,
    "S2": 99.76,
    "S3": 100.76,
    "S4": 105.32,
    "S5": 98.88
  },
  "transportation_costs": {
    "S1": {
      "C1": 150.74,
      "C2": 0.02,
      "C3": 49.13,
      "C4": 2080.15,
      "C5": 426.40
    },
    "S2": {
      "C1": 233.05,
      "C2": 97.73,
      "C3": 49.84,
      "C4": 1982.39,
      "C5": 23.96
    },
    "S3": {
      "C1": 55.68,
      "C2": 935.61,
      "C3": 4.03,
      "C4": 73.09,
      "C5": 525.32
    },
    "S4": {
      "C1": 1483.82,
      "C2": 1801.08,
      "C3": 112.16,
      "C4": 816.05,
      "C5": 107.01
    },
    "S5": {
      "C1": 1119.47,
      "C2": 884.31,
      "C3": 0.08,
      "C4": 1544.95,
      "C5": 543.67
    }
  },
  "demand": {
    "C1": 143,
    "C2": 6,
    "C3": 10,
    "C4": 25,
    "C5": 3
  }
}