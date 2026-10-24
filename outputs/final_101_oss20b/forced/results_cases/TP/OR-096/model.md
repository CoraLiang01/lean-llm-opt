**Mathematical model**

Let  

* \(S=\{I,II\}\) – set of elementary schools  
* \(N=\{N01,\dots ,N31\}\) – set of neighbourhoods  

Decision variables  

\[
\begin{aligned}
y_{s,n}^{W} &\ge 0 &&\text{white pupils from }n\text{ assigned to }s\\
y_{s,n}^{NW}&\ge 0 &&\text{non‑white pupils from }n\text{ assigned to }s
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
\text{Cap}[s] &\ge 0 &&\text{capacity of school }s\\
\text{PopW}[n] &\ge 0 &&\text{white population in neighbourhood }n\\
\text{PopNW}[n] &\ge 0 &&\text{non‑white population in neighbourhood }n\\
d_{s,n} &\ge 0 &&\text{distance (miles) from }s\text{ to }n
\end{aligned}
\]

---

### Objective
\[
\min \;\sum_{s\in S}\sum_{n\in N} d_{s,n}\,\bigl(y_{s,n}^{W}+y_{s,n}^{NW}\bigr)
\]

### Constraints  

1. **Population assignment**  
   \[
   \sum_{s\in S} y_{s,n}^{W}= \text{PopW}[n] \qquad \forall n\in N
   \]
   \[
   \sum_{s\in S} y_{s,n}^{NW}= \text{PopNW}[n] \qquad \forall n\in N
   \]

2. **School capacity**  
   \[
   \sum_{n\in N}\bigl(y_{s,n}^{W}+y_{s,n}^{NW}\bigr)\le \text{Cap}[s] \qquad \forall s\in S
   \]

3. **Racial balance (60 % ± 10 %)**  
   Let \(T_s=\sum_{n\in N}\bigl(y_{s,n}^{W}+y_{s,n}^{NW}\bigr)\) and  
   \(W_s=\sum_{n\in N} y_{s,n}^{W}\).  
   \[
   0.5\,T_s \;\le\; W_s \;\le\; 0.7\,T_s \qquad \forall s\in S
   \]

4. **Non‑negativity**  
   \[
   y_{s,n}^{W}\ge 0,\; y_{s,n}^{NW}\ge 0 \qquad \forall s\in S,\; n\in N
   \]

---

**Data‑mapping rules**

| Parameter | Source table | Column name | Notes |
|-----------|--------------|-------------|-------|
| `Cap[s]` | `file_0_view_0` | `Capacity` | School capacity |
| `PopW[n]` | `file_1_view_0` | `Population_White` | White population per neighbourhood |
| `PopNW[n]` | `file_1_view_0` | `Population_NonWhite` | Non‑white population per neighbourhood |
| `d_{s,n}` | `file_2_view_0` | `N01`–`N31` | Distance from school (row) to neighbourhood (column) |

All index sets (`S`, `N`) are taken directly from the identifiers present in the respective source tables. No additional aggregation or transformation is performed beyond the linear constraints listed above.