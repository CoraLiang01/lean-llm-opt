##### Index Sets  

- \(I\) : set of warehouses (indexed by the values in column **Unnamed: 0** of *fixed_cost.csv* and *transportation_costs.csv*).  
- \(J\) : set of customers (indexed by the values in column **customer** of *demand.csv*).  

##### Parameters  

- \(f_i\) : fixed operating cost of warehouse \(i\) (column **fixed_costs** of *fixed_cost.csv*).  
- \(c_{ij}\) : transportation cost per unit from warehouse \(i\) to customer \(j\) (columns **C1**–**C7** of *transportation_costs.csv*).  
- \(d_j\) : total demand of customer \(j\) (column **demand** of *demand.csv*).  

##### Decision Variables  

- \(y_i \in \{0,1\}\) : 1 if warehouse \(i\) is activated, 0 otherwise.  
- \(x_{ij} \ge 0\) : quantity shipped from warehouse \(i\) to customer \(j\).  

##### Objective Function  

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

##### Constraints  

###### 1. Demand Satisfaction  

\[
\sum_{i\in I} x_{ij} \;\ge\; d_j , \qquad \forall j\in J
\]

###### 2. Warehouse Activation (big‑M)  

\[
x_{ij} \;\le\; d_j\,y_i , \qquad \forall i\in I,\; \forall j\in J
\]

###### 3. Variable Domains  

\[
y_i \in \{0,1\} , \qquad \forall i\in I
\]

\[
x_{ij} \;\ge\; 0 , \qquad \forall i\in I,\; \forall j\in J
\]

##### Data Mapping  

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(d_j\)   | file_0_view_0   | demand        |
| \(f_i\)   | file_1_view_0   | fixed_costs   |
| \(c_{ij}\)| file_2_view_0   | C1–C7 (column *Ck* corresponds to customer \(j\)) |
| Index \(i\) | file_1_view_0 | Unnamed: 0   |
| Index \(j\) | file_0_view_0 | customer     |

All data are taken directly from the specified CSV files; no values or record counts are reproduced.