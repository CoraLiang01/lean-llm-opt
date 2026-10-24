**Mathematical model**

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I}x_{ij}\;\ge\;d_j \qquad\forall j\in J \quad\text{(demand)}\\
&\sum_{j\in J}x_{ij}\;\le\;s_i \qquad\forall i\in I \quad\text{(supply)}\\
&x_{ij}\;\ge\;0 \qquad\forall i\in I,\;j\in J
\end{aligned}
\]

*Sets*  
\(I=\{i\mid i\text{ is a supplier}\}\)  
\(J=\{j\mid j\text{ is a customer}\}\)

*Parameters*  
\(d_j\) – demand of customer \(j\)  
\(s_i\) – supply capacity of supplier \(i\)  
\(c_{ij}\) – unit transportation cost from supplier \(i\) to customer \(j\)

*Decision variables*  
\(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\)

---

**Data‑mapping rules**

| Parameter | Source table | Key column(s) | Value column(s) | Mapping rule |
|-----------|--------------|---------------|-----------------|--------------|
| \(d_j\) | `file_0_view_0` | `Customers` | `demand` | For each record \(r\): let \(j=r['Customers']\); set \(d_j=r['demand']\). |
| \(s_i\) | `file_1_view_0` | `Supplier` | `supply_capacity` | For each record \(r\): let \(i=r['Supplier']\); set \(s_i=r['supply_capacity']\). |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` (supplier id) | `demand1` … `demand8` (customer ids) | For each record \(r\): let \(i=r['Unnamed: 0']\); for each column \(k\in\{demand1,\dots,demand8\}\): let \(j\) be the customer id corresponding to column \(k\); set \(c_{ij}=r[k]\). |

*Index correspondence*  
- Supplier identifiers: `supplier1` … `supplier8` (from `file_1_view_0`) and `supply1` … `supply8` (from `file_2_view_0`) must match.  
- Customer identifiers: `demand1` … `demand8` (from `file_0_view_0`) correspond to the same column names in `file_2_view_0`.  

These mapping rules bind every parameter to the exact source data without any aggregation or omission.