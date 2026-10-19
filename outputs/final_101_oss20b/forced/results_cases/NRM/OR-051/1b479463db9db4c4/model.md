##### Index Sets  

- \(I\) : set of cabinets.  
- \(J\) : set of coffee products.  

##### Parameters  

- \(C_i\) : capacity of cabinet \(i\) (units of weight).  
- \(v_j\) : value of one unit of product \(j\).  
- \(w_j\) : weight of one unit of product \(j\).  

##### Variables  

- \(x_{ij}\) : number of units of product \(j\) placed in cabinet \(i\).  

##### Objective Function  

\[
\quad \quad \max \quad \sum_{i \in I}\sum_{j \in J} v_j \, x_{ij}
\]

##### Constraints  

###### 1. Weight Capacity Constraints  

\[
\quad \quad \sum_{j \in J} w_j \, x_{ij} \;\le\; C_i , \quad \forall i \in I
\]

###### 2. Variable Domain Constraints  

\[
\quad \quad x_{ij} \in \mathbb{Z}, \; x_{ij} \ge 0 , \quad \forall i \in I, \; \forall j \in J
\]

##### Data Mapping  

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(C_i\)   | file_0_view_0   | Capacity      |
| \(v_j\)   | file_1_view_0   | Value         |
| \(w_j\)   | file_1_view_0   | Weight        |