**Mathematical Model**

Let  

* \(S=\{I,II\}\) – set of elementary schools  
* \(N=\{N01,\dots ,N31\}\) – set of neighborhoods  

Parameters  

\[
\begin{aligned}
c_s &\quad\text{capacity of school }s\in S\\
p^{W}_n &\quad\text{white‑student population in neighborhood }n\in N\\
p^{NW}_n &\quad\text{non‑white‑student population in neighborhood }n\in N\\
d_{s,n} &\quad\text{distance (miles) from school }s\text{ to neighborhood }n
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x^{W}_{s,n} &\in \mathbb{Z}_{+}\quad\text{white pupils from }n\text{ assigned to }s\\
x^{NW}_{s,n} &\in \mathbb{Z}_{+}\quad\text{non‑white pupils from }n\text{ assigned to }s
\end{aligned}
\]

---

### Objective  

\[
\min \;\sum_{s\in S}\sum_{n\in N} d_{s,n}\,\bigl(x^{W}_{s,n}+x^{NW}_{s,n}\bigr)
\]

---

### Constraints  

1. **Neighborhood supply**

\[
\sum_{s\in S} x^{W}_{s,n}=p^{W}_n \quad\forall n\in N
\]
\[
\sum_{s\in S} x^{NW}_{s,n}=p^{NW}_n \quad\forall n\in N
\]

2. **School capacity**

\[
\sum_{n\in N}\bigl(x^{W}_{s,n}+x^{NW}_{s,n}\bigr)\le c_s \quad\forall s\in S
\]

3. **Racial balance (within ±10 % of 60 % white)**  

Let \(T_s=\sum_{n\in N}\bigl(x^{W}_{s,n}+x^{NW}_{s,n}\bigr)\) and \(W_s=\sum_{n\in N}x^{W}_{s,n}\).  
Then

\[
0.5\,T_s \;\le\; W_s \;\le\; 0.7\,T_s \quad\forall s\in S
\]

4. **Non‑negativity & integrality**

\[
x^{W}_{s,n},\,x^{NW}_{s,n}\in \mathbb{Z}_{+}\quad\forall s\in S,\;n\in N
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(c_s\) | `file_0_view_0` | `Capacity` | One row per school |
| \(p^{W}_n\) | `file_1_view_0` | `Population_White` | One row per neighborhood |
| \(p^{NW}_n\) | `file_1_view_0` | `Population_NonWhite` | One row per neighborhood |
| \(d_{s,n}\) | `file_2_view_0` | `N01`–`N31` | Each row corresponds to a school; columns are neighborhoods |

All index sets \(S\) and \(N\) are taken directly from the rows of the respective tables. No additional entities are introduced beyond those present in the data.