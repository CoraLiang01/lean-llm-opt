**Mathematical Model**

Let  

- \(I=\{1,\dots ,11\}\) : set of warehouses  
- \(J=\{1,\dots ,11\}\) : set of stores  

Parameters  

- \(f_i\) : opening cost of warehouse \(i\)  (\(i\in I\))  
- \(C_i\) : capacity of warehouse \(i\)  (\(i\in I\))  
- \(d_j\) : demand of store \(j\)  (\(j\in J\))  
- \(c_{ij}\) : transportation cost per unit from warehouse \(i\) to store \(j\) (\(i\in I,\;j\in J\))  

Decision variables  

- \(y_i\in\{0,1\}\) : 1 if warehouse \(i\) is opened, 0 otherwise  
- \(x_{ij}\ge 0\) : units shipped from warehouse \(i\) to store \(j\)

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} = d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij} \le C_i\,y_i \qquad &&\forall i\in I \quad\text{(capacity)}\\[4pt]
& x_{ij}\ge 0 \qquad &&\forall i\in I,\;j\in J\\
& y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Index set | Source | Column(s) | Mapping rule |
|-----------|--------|-----------|--------------|
| \(I\) | `file_0_view_0` | `Warehouse (i)` | All 11 rows; values 1–11 |
| \(J\) | `file_1_view_0` | `Store (j)` | All 11 rows; values 1–11 |
| \(f_i\) | `file_0_view_0` | `Opening Cost (fi)` | Map by `Warehouse (i)` |
| \(C_i\) | `file_0_view_0` | `Capacity (units)` | Map by `Warehouse (i)` |
| \(d_j\) | `file_1_view_0` | `Demand (units, dj)` | Map by `Store (j)` |
| \(c_{ij}\) | `file_2_view_0` | Matrix cell at row `Unnamed: 0` = store \(j\), column `Wk` = warehouse \(i\) | Map row and column identifiers to indices \(j\) and \(i\) respectively |

All parameters are taken directly from the corresponding CSV files; no additional processing or filtering is applied. The model uses the full sets of warehouses and stores as provided.