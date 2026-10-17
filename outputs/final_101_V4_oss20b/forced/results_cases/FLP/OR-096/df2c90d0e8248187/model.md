**Symbolic Model**

Let  

- \(S\) – set of schools (indices \(s\)).  
- \(N\) – set of neighborhoods (indices \(n\)).  

Parameters  

- \(cap_s\) – capacity of school \(s\).  
- \(popW_n\) – white‑student population in neighborhood \(n\).  
- \(popNW_n\) – non‑white‑student population in neighborhood \(n\).  
- \(d_{sn}\) – distance (miles) from school \(s\) to neighborhood \(n\).  

Decision variables  

- \(x_{sn}^W \ge 0\) – number of white students from neighborhood \(n\) assigned to school \(s\).  
- \(x_{sn}^{NW} \ge 0\) – number of non‑white students from neighborhood \(n\) assigned to school \(s\).  

Objective  

\[
\min \sum_{s\in S}\sum_{n\in N} d_{sn}\,\bigl(x_{sn}^W+x_{sn}^{NW}\bigr)
\]

Constraints  

1. **Neighborhood population balance**  
   \[
   \sum_{s\in S} x_{sn}^W = popW_n ,\qquad
   \sum_{s\in S} x_{sn}^{NW} = popNW_n ,\quad \forall n\in N
   \]

2. **School capacity**  
   \[
   \sum_{n\in N}\bigl(x_{sn}^W+x_{sn}^{NW}\bigr) \le cap_s ,\quad \forall s\in S
   \]

3. **Racial‑balance limits**  
   Let \(W_s=\sum_{n}x_{sn}^W\) and \(T_s=\sum_{n}(x_{sn}^W+x_{sn}^{NW})\).  
   \[
   0.5\,T_s \;\le\; W_s \;\le\; 0.7\,T_s ,\quad \forall s\in S
   \]

All variables are non‑negative continuous.

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column | Mapping Rule |
|-----------|--------------|------------|--------------|--------------|
| \(cap_s\) | `file_0_view_0` | `School` | `Capacity` | `cap_s = value where School = s` |
| \(popW_n\) | `file_1_view_0` | `Neighborhood` | `Population_White` | `popW_n = value where Neighborhood = n` |
| \(popNW_n\) | `file_1_view_0` | `Neighborhood` | `Population_NonWhite` | `popNW_n = value where Neighborhood = n` |
| \(d_{sn}\) | `file_2_view_0` | `School` (row) | `N01`–`N31` (columns) | `d_{sn} = value where School = s and column = n` |

All index sets \(S\) and \(N\) are taken from the distinct values of the key columns in the respective tables. No additional data or constraints are introduced beyond those specified.