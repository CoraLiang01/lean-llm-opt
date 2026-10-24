**Mathematical Model**

Let  

* \(I=\{\,\text{Supplier1},\dots ,\text{Supplier5}\,\}\) – set of warehouses  
* \(J=\{\,\text{Customer1},\dots ,\text{Customer6}\,\}\) – set of retail stores  

Parameters  

* \(d_j\) – daily demand of store \(j\)  \(j\in J\)  
* \(s_i\) – supply capacity of warehouse \(i\)  \(i\in I\)  
* \(c_{ij}\) – transportation cost per unit from warehouse \(i\) to store \(j\)  \(i\in I,\;j\in J\)  

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from warehouse \(i\) to store \(j\)  \(i\in I,\;j\in J\)

---

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\le s_i , &&\forall i\in I \quad\text{(capacity)}\\[4pt]
&x_{ij}\ge 0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| \(d_j\) | `file_0_view_0` | `demand` | Row identified by `Customers` |
| \(s_i\) | `file_1_view_0` | `supply_capacity` | Row identified by `Suppliers` |
| \(c_{ij}\) | `file_2_view_0` | matrix cells | Row identified by `Unnamed: 0` (Suppliers), column by `Customer1`–`Customer6` |

All index sets \(I\) and \(J\) are taken from the complete lists of suppliers and customers returned in the observation. No additional constraints or variables are introduced.