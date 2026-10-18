##### Index Sets  

$ I \;=\; \{\,\text{S1},\text{S2},\text{S3},\text{S4},\text{S5},\text{S6},\text{S7},\text{S8}\,\}$  

$ J \;=\; \{\,\text{C1},\text{C2},\text{C3},\text{C4},\text{C5},\text{C6},\text{C7},\text{C8},\text{C9}\,\}$  

##### Parameters  

$ f_i \;=\; \text{fixed cost of supplier } i \quad (i\in I)$  

$ t_{ij} \;=\; \text{transportation cost per vehicle from supplier } i \text{ to customer } j \quad (i\in I,\;j\in J)$  

$ d_j \;=\; \text{demand of customer } j \quad (j\in J)$  

##### Variables  

$ y_i \;\in\;\{0,1\} \quad (i\in I)$  

$ x_{ij} \;\ge\; 0 \quad (i\in I,\;j\in J)$  

##### Objective Function  

$\displaystyle \min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}$  

##### Constraints  

###### 1. Demand Satisfaction  

$\displaystyle \sum_{i\in I} x_{ij} \;\ge\; d_j , \quad \forall j\in J$  

###### 2. Supplier Activation Linking  

$\displaystyle x_{ij} \;\le\; d_j\,y_i , \quad \forall i\in I,\; j\in J$  

###### 3. Non‑negativity  

$\displaystyle x_{ij} \;\ge\; 0 , \quad \forall i\in I,\; j\in J$  

###### 4. Binary Supplier Variables  

$\displaystyle y_i \;\in\;\{0,1\} , \quad \forall i\in I$  

##### Data Mapping  

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| $f_i$ | file_1_view_0 | fixed_costs |
| $t_{ij}$ | file_2_view_0 | C1 … C9 (columns per customer) |
| $d_j$ | file_0_view_0 | demand |

All data values are taken directly from the corresponding CSV files; no record values or counts are reproduced here.