**Mathematical Model**

Let  

* \(I=\{i\mid i\text{ is a supplier}\}\)  
* \(J=\{j\mid j\text{ is a customer}\}\)

Parameters  

* \(d_j\) – daily demand of customer \(j\)  
* \(s_i\) – daily supply capacity of supplier \(i\)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\)

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\le s_i , &&\forall i\in I \quad\text{(supply capacity)}\\[4pt]
&x_{ij}\ge 0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(d_j\)   | `file_0_view_0` | `Customers` | `demand` |
| \(s_i\)   | `file_1_view_0` | `Supplier` | `supply_capacity` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` (supplier) | `demand1`–`demand8` (customers) |

*Index sets*  
- \(I\) is the set of distinct values in the `Supplier` column of `file_1_view_0`.  
- \(J\) is the set of distinct values in the `Customers` column of `file_0_view_0`.  

All parameters are mapped directly from the corresponding CSV columns; no numerical values are enumerated in the model.