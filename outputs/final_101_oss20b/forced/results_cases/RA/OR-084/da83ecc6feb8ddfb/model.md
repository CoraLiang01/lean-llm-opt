##### Objective Function: 

$\quad \quad \min \; C_{\max}$

##### Constraints

###### 1. Assignment Constraints: 

$\quad \quad \sum_{j=1}^{3} x_{ij} = 1, \quad \forall i \in \{1,\dots ,40\}$

###### 2. CPU Capacity Constraints: 

$\quad \quad \sum_{i=1}^{40} \frac{BI_i}{f_j} \, x_{ij} \leq C_{\max}, \quad \forall j \in \{1,2,3\}$

###### 3. Non‑negativity and Integrality Constraints: 

$\quad \quad x_{ij} \in \{0,1\}, \quad \forall i \in \{1,\dots ,40\}, \; \forall j \in \{1,2,3\}$  

$\quad \quad C_{\max} \geq 0$

*Here $BI_i$ denotes the basic instruction count (in billions) for task $i$ as given in the data, and $f_j$ is the frequency of CPU $j$ (1.33, 2, and 2.66 GHz respectively).*

###### Retrieved Information

{
  "tasks": [
    {"Task": 1, "BI": 1.1},
    {"Task": 2, "BI": 2.1},
    {"Task": 3, "BI": 3.0},
    {"Task": 4, "BI": 1.0},
    {"Task": 5, "BI": 0.7},
    {"Task": 6, "BI": 5.0},
    {"Task": 7, "BI": 3.0},
    {"Task": 8, "BI": 3.5},
    {"Task": 9, "BI": 4.4},
    {"Task": 10, "BI": 3.8},
    {"Task": 11, "BI": 3.5},
    {"Task": 12, "BI": 2.8},
    {"Task": 13, "BI": 4.1},
    {"Task": 14, "BI": 2.9},
    {"Task": 15, "BI": 5.4},
    {"Task": 16, "BI": 5.8},
    {"Task": 17, "BI": 2.6},
    {"Task": 18, "BI": 4.9},
    {"Task": 19, "BI": 3.4},
    {"Task": 20, "BI": 3.6},
    {"Task": 21, "BI": 5.6},
    {"Task": 22, "BI": 0.9},
    {"Task": 23, "BI": 1.0},
    {"Task": 24, "BI": 0.6},
    {"Task": 25, "BI": 5.1},
    {"Task": 26, "BI": 4.8},
    {"Task": 27, "BI": 5.3},
    {"Task": 28, "BI": 5.9},
    {"Task": 29, "BI": 4.9},
    {"Task": 30, "BI": 3.0},
    {"Task": 31, "BI": 4.8},
    {"Task": 32, "BI": 1.2},
    {"Task": 33, "BI": 4.0},
    {"Task": 34, "BI": 1.3},
    {"Task": 35, "BI": 5.7},
    {"Task": 36, "BI": 3.4},
    {"Task": 37, "BI": 2.8},
    {"Task": 38, "BI": 2.0},
    {"Task": 39, "BI": 4.8},
    {"Task": 40, "BI": 3.0}
  ],
  "CPUs": [
    {"CPU": 1, "Frequency_GHz": 1.33},
    {"CPU": 2, "Frequency_GHz": 2.0},
    {"CPU": 3, "Frequency_GHz": 2.66}
  ]
}