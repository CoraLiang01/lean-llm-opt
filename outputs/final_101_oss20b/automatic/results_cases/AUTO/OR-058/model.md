**Mathematical Model**

Let  

* \(I=\{i\mid i\text{ is a supplier in } \text{file\_1\_view\_0}\}\)  
* \(J=\{j\mid j\text{ is a store in } \text{file\_0\_view\_0}\}\)

Parameters  

* \(f_i\) – fixed cost of opening supplier \(i\)  
* \(c_{ij}\) – unit transportation cost from supplier \(i\) to store \(j\)  
* \(d_j\) – demand of store \(j\)  

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to store \(j\)

Let  

\[
M=\sum_{j\in J} d_j
\]

---

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

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C6` (columns) |
| \(d_j\)   | `file_0_view_0` | `customer` | `demand` |

*Index sets*  
- \(I\) is the set of values in `file_1_view_0.Unnamed: 0`.  
- \(J\) is the set of values in `file_0_view_0.customer`.  

*Big‑M*  
\(M\) is computed as the sum of all \(d_j\) values from `file_0_view_0`.