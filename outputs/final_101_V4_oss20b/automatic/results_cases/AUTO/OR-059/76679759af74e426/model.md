**Mathematical Model**

Let  

* \(I=\{i\}\) – set of suppliers (rows of *fixed_cost.csv*).  
* \(J=\{j\}\) – set of dealerships (rows of *demand.csv*).  

Parameters  

* \(f_i\) – fixed opening cost of supplier \(i\).  
* \(d_j\) – demand of dealership \(j\).  
* \(c_{ij}\) – transportation cost per vehicle from supplier \(i\) to dealership \(j\).  

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.  
* \(x_{ij}\ge 0\) – number of vehicles shipped from supplier \(i\) to dealership \(j\).  

Let \(M=\sum_{j\in J} d_j\) (a valid upper bound on total shipments).

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}=d_j , && \forall j\in J \quad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij}\le M\,y_i , && \forall i\in I \quad\text{(activation)}\\[4pt]
& x_{ij}\ge 0 , && \forall i\in I,\; j\in J\\
& y_i\in\{0,1\} , && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Index Column | Value Column |
|-----------|--------------|--------------|--------------|
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(d_j\)   | `file_0_view_0` | `customer`   | `demand`      |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` | `C1`–`C9` (column matching dealership `j`) |

*Index sets*  
- \(I\) is the set of all values in `file_1_view_0`’s `Unnamed: 0`.  
- \(J\) is the set of all values in `file_0_view_0`’s `customer`.  

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.