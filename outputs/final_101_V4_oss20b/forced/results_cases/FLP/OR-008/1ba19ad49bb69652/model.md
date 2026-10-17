**Mathematical Model**

Let  

- \(I=\{\text{Supplier1},\dots ,\text{Supplier5}\}\)  
- \(J=\{\text{Customer1},\dots ,\text{Customer6}\}\)

Parameters  

- \(d_j\) : daily demand of customer \(j\)  \(j\in J\)  
- \(s_i\) : supply capacity of supplier \(i\) \(i\in I\)  
- \(c_{ij}\) : transportation cost per unit from supplier \(i\) to customer \(j\) \(i\in I,\;j\in J\)

Decision variables  

- \(x_{ij}\ge 0\) : quantity shipped from supplier \(i\) to customer \(j\) \(i\in I,\;j\in J\)

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
| \(c_{ij}\) | `file_2_view_0` | `Customer1`–`Customer6` | Row identified by `Unnamed: 0` (Suppliers) |

All indices \(i\) and \(j\) are taken from the full set of rows/columns returned in the observation. No additional constraints or parameters are introduced.