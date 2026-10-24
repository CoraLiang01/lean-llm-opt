**Mathematical Model**

Let  

* \(S=\{I,II\}\) – set of elementary schools.  
* \(N=\{N01,\dots ,N31\}\) – set of neighborhoods.  

Decision variables  

\[
\begin{aligned}
x_{s,n}^{W} &\in \mathbb{Z}_{\ge 0} &&\text{white pupils from }n\text{ assigned to }s,\\
x_{s,n}^{NW}&\in \mathbb{Z}_{\ge 0} &&\text{non‑white pupils from }n\text{ assigned to }s,\\
T_s &\in \mathbb{Z}_{\ge 0} &&\text{total pupils assigned to }s.
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
\text{Cap}_s &\ge 0 &&\text{capacity of school }s,\\
\text{Pop}^{W}_n &\ge 0 &&\text{white population in neighborhood }n,\\
\text{Pop}^{NW}_n &\ge 0 &&\text{non‑white population in neighborhood }n,\\
d_{s,n} &\ge 0 &&\text{distance (miles) from }s\text{ to }n.
\end{aligned}
\]

---

### Objective
\[
\min \;\sum_{s\in S}\sum_{n\in N} d_{s,n}\,\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)
\]

### Constraints  

1. **Neighborhood supply**  
   \[
   \sum_{s\in S} x_{s,n}^{W}= \text{Pop}^{W}_n \quad \forall n\in N
   \]
   \[
   \sum_{s\in S} x_{s,n}^{NW}= \text{Pop}^{NW}_n \quad \forall n\in N
   \]

2. **School capacity**  
   \[
   \sum_{n\in N}\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)\le \text{Cap}_s \quad \forall s\in S
   \]

3. **Total pupils per school**  
   \[
   T_s=\sum_{n\in N}\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr) \quad \forall s\in S
   \]

4. **Racial‑balance (60 % ± 10 %)**  
   \[
   0.5\,T_s \;\le\; \sum_{n\in N} x_{s,n}^{W}\;\le\;0.7\,T_s \quad \forall s\in S
   \]

5. **Non‑negativity & integrality**  
   \[
   x_{s,n}^{W},\,x_{s,n}^{NW},\,T_s \in \mathbb{Z}_{\ge 0}\quad \forall s\in S,\;n\in N
   \]

---

## Data Mapping

| Parameter | Source Table | Column | Key |
|-----------|--------------|--------|-----|
| \(\text{Cap}_s\) | `file_0_view_0` | `Capacity` | `School` |
| \(\text{Pop}^{W}_n\) | `file_1_view_0` | `Population_White` | `Neighborhood` |
| \(\text{Pop}^{NW}_n\) | `file_1_view_0` | `Population_NonWhite` | `Neighborhood` |
| \(d_{s,n}\) | `file_2_view_0` | `N01`–`N31` | `School` (row), `Neighborhood` (column) |

All index sets \(S\) and \(N\) are taken directly from the distinct values of the `School` and `Neighborhood` columns in the respective tables. No additional data are introduced beyond what is supplied in the observation.