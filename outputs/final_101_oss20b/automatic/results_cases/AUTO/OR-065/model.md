**Mathematical Model**

Let  

* \(I=\{S1,S2,S3\}\) (warehouses)  
* \(J=\{C1,C2,C3\}\) (customers)

Parameters  

* \(d_j\) – demand of customer \(j\)  \(j\in J\)  
* \(f_i\) – fixed cost of opening warehouse \(i\)  \(i\in I\)  
* \(c_{ij}\) – unit transportation cost from warehouse \(i\) to customer \(j\) \(i\in I,\;j\in J\)  
* \(M=\sum_{j\in J} d_j\) (upper bound on total shipment from any warehouse)

Decision variables  

* \(x_{ij}\ge 0\) quantity shipped from warehouse \(i\) to customer \(j\) \(i\in I,\;j\in J\)  
* \(y_i\in\{0,1\}\) indicator that warehouse \(i\) is activated \(i\in I\)

\[
\begin{aligned}
\min\;& \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}= d_j , &&\forall j\in J \quad\text{(demand satisfaction)}\\[4pt]
& \sum_{j\in J} x_{ij}\le M\,y_i , &&\forall i\in I \quad\text{(activation constraint)}\\[4pt]
& x_{ij}\ge 0 , &&\forall i\in I,\;j\in J\\
& y_i\in\{0,1\} , &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column | Mapping |
|-----------|--------------|------------|--------------|---------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | \(d_j =\) value where `customer` = \(j\) |
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | \(f_i =\) value where `Unnamed: 0` = \(i\) |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` (row) | column `Cj` | \(c_{ij} =\) value where row `Unnamed: 0` = \(i\) and column `Cj` = \(j\) |
| \(M\) | – | – | – | \(M = \sum_{j\in J} d_j\) (computed from \(d_j\)) |

All indices \(i\) and \(j\) are taken from the complete set of identifiers present in the respective source tables. No additional capacity limits are imposed beyond the big‑\(M\) activation constraint.