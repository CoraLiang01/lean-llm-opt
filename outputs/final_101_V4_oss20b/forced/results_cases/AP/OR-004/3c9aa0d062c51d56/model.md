**Mathematical Model**

Let  

* \(S\) – set of distribution centers (rows of *transportation_costs.csv*).  
* \(C\) – set of customer groups (columns of *transportation_costs.csv*).  

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{daily demand of customer }c \quad (c\in C)\\
s_s &\;=\;\text{supply capacity of center }s \quad (s\in S)\\
c_{s,c} &\;=\;\text{transportation cost per unit from }s\text{ to }c \quad (s\in S,\;c\in C)
\end{aligned}
\]

Decision variables  

\[
x_{s,c}\;\ge 0 \qquad (s\in S,\;c\in C)
\]

*Objective*  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

*Constraints*  

1. **Supply capacity**  
\[
\sum_{c\in C} x_{s,c}\;\le\; s_s \qquad \forall s\in S
\]

2. **Demand satisfaction**  
\[
\sum_{s\in S} x_{s,c}\;=\; d_c \qquad \forall c\in C
\]

3. **Non‑negativity**  
\[
x_{s,c}\;\ge\;0 \qquad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Mapping Rule |
|-----------|--------------|------------|--------------|--------------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` | For each row, map `customer` → `c`, `demand` → `d_c`. |
| \(s_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | For each row, map `Unnamed: 0` → `s`, `supply_capacity` → `s_s`. |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (row id) → `s` | `C1`–`C12` (columns) → `c` | For each cell, map row id `Unnamed: 0` → `s`, column header `Ck` → `c`, value → `c_{s,c}`. |

*All index sets \(S\) and \(C\) are the full sets of row and column identifiers extracted from the respective tables.*