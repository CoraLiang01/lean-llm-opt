**Mathematical Model**

Let  

- \(I\) be the set of warehouses (identified by the values in the column **Warehouse (i)**).  
- \(J\) be the set of stores (identified by the values in the column **Store (j)**).  

Parameters  

- \(f_i\) – opening cost of warehouse \(i\).  
- \(C_i\) – capacity of warehouse \(i\).  
- \(d_j\) – demand of store \(j\).  
- \(c_{ij}\) – transportation cost per unit from warehouse \(i\) to store \(j\).  

Decision variables  

- \(y_i \in \{0,1\}\) – 1 if warehouse \(i\) is opened, 0 otherwise.  
- \(x_{ij} \in \mathbb{Z}_{\ge 0}\) – number of units shipped from warehouse \(i\) to store \(j\).  

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} \;=\; d_j , && \forall j\in J \quad\text{(demand satisfaction)}\\[4pt]
& \sum_{j\in J} x_{ij} \;\le\; C_i\,y_i , && \forall i\in I \quad\text{(capacity)}\\[4pt]
& x_{ij} \;\ge\; 0 , && \forall i\in I,\; j\in J \quad\text{(non‑negativity)}\\[4pt]
& y_i \;\in\; \{0,1\} , && \forall i\in I \quad\text{(binary)}\\
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | ID Column | Value Column |
|-----------|--------------|-----------|--------------|
| \(f_i\) | `file_0_view_0` | `Warehouse (i)` | `Opening Cost (fi)` |
| \(C_i\) | `file_0_view_0` | `Warehouse (i)` | `Capacity (units)` |
| \(d_j\) | `file_1_view_0` | `Store (j)` | `Demand (units, dj)` |
| \(c_{ij}\) | `file_2_view_0` | Row ID: `Unnamed: 0` (store) | Column ID: `Warehouse (i)` (warehouse) |

The sets \(I\) and \(J\) are the distinct values appearing in the respective ID columns of the source tables. All numerical values are taken directly from the CSV files; no additional data are introduced.