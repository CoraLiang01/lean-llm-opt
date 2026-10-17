**Mathematical Model**

Let  

* \(S=\{I,II\}\) – set of schools (from *file_0_view_0*).  
* \(N=\{N01,\dots ,N31\}\) – set of neighborhoods (from *file_1_view_0*).  

Parameters  

* \(C_s\) – capacity of school \(s\) (column **Capacity** in *file_0_view_0*).  
* \(W_n\) – number of white pupils in neighborhood \(n\) (column **Population_White** in *file_1_view_0*).  
* \(NW_n\) – number of non‑white pupils in neighborhood \(n\) (column **Population_NonWhite** in *file_1_view_0*).  
* \(d_{s,n}\) – distance from school \(s\) to neighborhood \(n\) (columns **N01**–**N31** in *file_2_view_0*).  

Decision variables  

\[
x_{s,n,w}\in \mathbb{Z}_{\ge 0}\quad\text{(white pupils from }n\text{ assigned to }s)
\]
\[
x_{s,n,nw}\in \mathbb{Z}_{\ge 0}\quad\text{(non‑white pupils from }n\text{ assigned to }s)
\]

---

**Objective**

\[
\min \;\sum_{s\in S}\sum_{n\in N} d_{s,n}\,\bigl(x_{s,n,w}+x_{s,n,nw}\bigr)
\]

---

**Constraints**

1. **Neighborhood supply**

\[
\sum_{s\in S} x_{s,n,w}=W_n \quad\forall n\in N
\]
\[
\sum_{s\in S} x_{s,n,nw}=NW_n \quad\forall n\in N
\]

2. **School capacity**

\[
\sum_{n\in N}\bigl(x_{s,n,w}+x_{s,n,nw}\bigr)\le C_s \quad\forall s\in S
\]

3. **Racial balance (50 %–70 % white)**  

Let \(T_s=\sum_{n}(x_{s,n,w}+x_{s,n,nw})\) and \(W_s=\sum_{n}x_{s,n,w}\).  
Then

\[
0.5\,T_s \;\le\; W_s \;\le\; 0.7\,T_s \quad\forall s\in S
\]

4. **Non‑negativity & integrality**

\[
x_{s,n,w},\,x_{s,n,nw}\in \mathbb{Z}_{\ge 0}\quad\forall s\in S,\;n\in N
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(C_s\) | `file_0_view_0` | `Capacity` |
| \(W_n\) | `file_1_view_0` | `Population_White` |
| \(NW_n\) | `file_1_view_0` | `Population_NonWhite` |
| \(d_{s,n}\) | `file_2_view_0` | `N01`–`N31` (each column corresponds to a neighborhood) |

All index sets \(S\) and \(N\) are taken directly from the respective tables. No additional data are introduced.