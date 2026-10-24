**Mathematical model**

Let  

* \(I=\{\,\text{Supplier1},\dots ,\text{Supplier5}\,\}\)  
* \(J=\{\,\text{Customer1},\dots ,\text{Customer6}\,\}\)

Parameters  

* \(d_j\) – daily demand of customer \(j\)  (\(j\in J\))  
* \(s_i\) – supply capacity of supplier \(i\)  (\(i\in I\))  
* \(c_{ij}\) – unit transportation cost from supplier \(i\) to customer \(j\)  (\(i\in I,\;j\in J\))

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\)  (\(i\in I,\;j\in J\))

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}\;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\;j\in J .
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Source column | Mapping rule |
|-----------|--------------|---------------|--------------|
| \(d_j\) | `file_0_view_0` | `demand` | For each row, map `Customers` → \(j\); value → \(d_j\). |
| \(s_i\) | `file_1_view_0` | `supply_capacity` | For each row, map `Suppliers` → \(i\); value → \(s_i\). |
| \(c_{ij}\) | `file_2_view_0` | matrix cells | Row axis: `Unnamed: 0` → \(i\); column axis: `Customers` → \(j\); cell value → \(c_{ij}\). |

| Index set | Source table | Source column | Mapping rule |
|-----------|--------------|---------------|--------------|
| \(I\) | `file_1_view_0` | `Suppliers` | Each distinct value → an element of \(I\). |
| \(J\) | `file_0_view_0` | `Customers` | Each distinct value → an element of \(J\). |

All numeric values are taken directly from the corresponding CSV files; no aggregation or transformation is performed beyond the mapping above.