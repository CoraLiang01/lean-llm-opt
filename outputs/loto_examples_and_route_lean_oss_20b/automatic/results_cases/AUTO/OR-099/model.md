**Mathematical Model**

Let  

* \(I=\{1,\dots ,11\}\) – set of warehouses  
* \(J=\{1,\dots ,11\}\) – set of stores  

Parameters  

* \(f_i\) – opening cost of warehouse \(i\)  
* \(C_i\) – capacity of warehouse \(i\)  
* \(d_j\) – demand of store \(j\)  
* \(c_{ij}\) – transportation cost per unit from warehouse \(i\) to store \(j\)

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if warehouse \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – units shipped from warehouse \(i\) to store \(j\)

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}=d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij}\le C_i\,y_i \qquad &&\forall i\in I \quad\text{(capacity)}\\[4pt]
& x_{ij}\ge 0 \qquad &&\forall i\in I,\;j\in J\\
& y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(f_i\) | `file_0_view_0` | `Opening Cost (fi)` | \(f_i =\) value in row where `Warehouse (i)=i\) |
| \(C_i\) | `file_0_view_0` | `Capacity (units)` | \(C_i =\) value in row where `Warehouse (i)=i\) |
| \(d_j\) | `file_1_view_0` | `Demand (units, dj)` | \(d_j =\) value in row where `Store (j)=j\) |
| \(c_{ij}\) | `file_2_view_0` | matrix entry at row `Unnamed: 0=j` and column `W{i}` | \(c_{ij} =\) value in that cell |

*Index sets*  
\(I=\{1,\dots ,11\}\) (warehouse identifiers from `Warehouse (i)`)  
\(J=\{1,\dots ,11\}\) (store identifiers from `Store (j)`)

All parameters are taken directly from the corresponding CSV files; no additional data or assumptions are introduced.