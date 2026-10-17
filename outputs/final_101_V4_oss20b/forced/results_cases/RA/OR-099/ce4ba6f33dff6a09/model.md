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
* \(x_{ij}\in\mathbb{Z}_{\ge 0}\) – units shipped from warehouse \(i\) to store \(j\)

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} = d_j , && \forall j\in J \quad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij} \le C_i\,y_i , && \forall i\in I \quad\text{(capacity)}\\[4pt]
& x_{ij}\ge 0,\; x_{ij}\in\mathbb{Z}, && \forall i\in I,\; j\in J\\[2pt]
& y_i\in\{0,1\}, && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Key Column | Value Column |
|-----------|----------|------------|--------------|
| \(f_i\) | `file_0_view_0` | `Warehouse (i)` | `Opening Cost (fi)` |
| \(C_i\) | `file_0_view_0` | `Warehouse (i)` | `Capacity (units)` |
| \(d_j\) | `file_1_view_0` | `Store (j)` | `Demand (units, dj)` |
| \(c_{ij}\) | `file_2_view_0` | Row key: `Unnamed: 0` (store id `Wj`) | Column key: `Warehouse (i)` (warehouse id `Wi`) |

*The mapping for \(c_{ij}\) uses the matrix structure of `file_2_view_0`: each row corresponds to a store (identified by `Unnamed: 0`), each column corresponds to a warehouse (identified by `Warehouse (i)`), and the cell value is the transportation cost.*