**Mathematical Model**

Let  

* \(S=\{I,\,II\}\) – set of elementary schools  
* \(N=\{N01,\dots ,N31\}\) – set of neighborhoods  

Decision variables  

\[
\begin{aligned}
y_{s,n}^{W} &\ge 0 &&\text{(white pupils from }n\text{ assigned to }s)\\
y_{s,n}^{NW}&\ge 0 &&\text{(non‑white pupils from }n\text{ assigned to }s)
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
\text{Cap}_s &\quad &&\text{capacity of school }s\\
\text{Pop}^{W}_n &\quad &&\text{white population in neighborhood }n\\
\text{Pop}^{NW}_n &\quad &&\text{non‑white population in neighborhood }n\\
d_{s,n} &\quad &&\text{distance (miles) from school }s\text{ to neighborhood }n
\end{aligned}
\]

---

### Objective  

\[
\min \;\sum_{s\in S}\sum_{n\in N} d_{s,n}\,\bigl(y_{s,n}^{W}+y_{s,n}^{NW}\bigr)
\]

---

### Constraints  

1. **Neighborhood supply**  
   \[
   \sum_{s\in S} y_{s,n}^{W}= \text{Pop}^{W}_n \quad \forall n\in N
   \]
   \[
   \sum_{s\in S} y_{s,n}^{NW}= \text{Pop}^{NW}_n \quad \forall n\in N
   \]

2. **School capacity**  
   \[
   \sum_{n\in N}\bigl(y_{s,n}^{W}+y_{s,n}^{NW}\bigr)\le \text{Cap}_s
   \quad \forall s\in S
   \]

3. **Racial balance (60 % ± 10 %)**  
   Let \(T_s=\sum_{n\in N}\bigl(y_{s,n}^{W}+y_{s,n}^{NW}\bigr)\).  
   \[
   0.5\,T_s \;\le\; \sum_{n\in N} y_{s,n}^{W}\;\le\; 0.7\,T_s
   \quad \forall s\in S
   \]

4. **Non‑negativity**  
   \[
   y_{s,n}^{W}\ge 0,\; y_{s,n}^{NW}\ge 0 \quad \forall s\in S,\; n\in N
   \]

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(\text{Cap}_s\) | `file_0_view_0` | `Capacity` | `School` → `s` |
| \(\text{Pop}^{W}_n\) | `file_1_view_0` | `Population_White` | `Neighborhood` → `n` |
| \(\text{Pop}^{NW}_n\) | `file_1_view_0` | `Population_NonWhite` | `Neighborhood` → `n` |
| \(d_{s,n}\) | `file_2_view_0` | `N01`–`N31` | `School` → `s`, column `Nxx` → `n` |

All parameters are taken directly from the corresponding CSV files; no aggregation or transformation beyond the mappings above is performed. The model uses continuous non‑negative variables; if integer solutions are required, declare \(y_{s,n}^{W}, y_{s,n}^{NW}\) as integers.