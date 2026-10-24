**Mathematical Model**

Let  

* \(S=\{I,II\}\) – set of schools  
* \(N=\{N01,\dots ,N31\}\) – set of neighborhoods  

Parameters  

* \(c_{s}\) – capacity of school \(s\)  (\(s\in S\))  
* \(w_{n}\) – number of white pupils in neighborhood \(n\)  (\(n\in N\))  
* \(nw_{n}\) – number of non‑white pupils in neighborhood \(n\)  (\(n\in N\))  
* \(d_{s,n}\) – distance (miles) from school \(s\) to neighborhood \(n\)  (\(s\in S,n\in N\))  

Decision variables  

* \(x_{s,n}^{W}\ge 0\) – white pupils from \(n\) assigned to \(s\)  
* \(x_{s,n}^{NW}\ge 0\) – non‑white pupils from \(n\) assigned to \(s\)  

---

### Objective  

\[
\min \;\sum_{s\in S}\sum_{n\in N} d_{s,n}\,\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)
\]

---

### Constraints  

1. **Neighborhood population balance**  
   \[
   \sum_{s\in S}x_{s,n}^{W}=w_{n}\qquad \forall n\in N
   \]
   \[
   \sum_{s\in S}x_{s,n}^{NW}=nw_{n}\qquad \forall n\in N
   \]

2. **School capacity**  
   \[
   \sum_{n\in N}\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)\le c_{s}\qquad \forall s\in S
   \]

3. **Racial balance (60 % ± 10 %)**  
   Let \(T_{s}=\sum_{n\in N}\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)\).  
   \[
   0.5\,T_{s}\;\le\;\sum_{n\in N}x_{s,n}^{W}\;\le\;0.7\,T_{s}\qquad \forall s\in S
   \]

4. **Non‑negativity**  
   \[
   x_{s,n}^{W}\ge 0,\;x_{s,n}^{NW}\ge 0\qquad \forall s\in S,\;n\in N
   \]

---

**Data Mapping**

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| \(c_{s}\) | `file_0_view_0` | `School`, `Capacity` |
| \(w_{n}\) | `file_1_view_0` | `Neighborhood`, `Population_White` |
| \(nw_{n}\) | `file_1_view_0` | `Neighborhood`, `Population_NonWhite` |
| \(d_{s,n}\) | `file_2_view_0` | `School`, `N01`–`N31` (each column is a neighborhood) |

All indices \(s\) and \(n\) are taken directly from the identifiers in the respective tables. No additional aggregation or transformation is performed beyond the linear constraints shown above.