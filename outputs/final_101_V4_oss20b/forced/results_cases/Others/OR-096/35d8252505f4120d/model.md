## Mathematical Model

**Sets**  
\(S=\{I,\,II\}\) (schools)  
\(N=\{N01,\dots ,N31\}\) (neighborhoods)

**Parameters**  

| Parameter | Symbol | Source | Column(s) |
|-----------|--------|--------|-----------|
| School capacity | \(C_s\) | `school_capacity.csv` | `Capacity` |
| White population in neighborhood | \(W_n\) | `neighborhoods_population.csv` | `Population_White` |
| Non‑white population in neighborhood | \(NW_n\) | `neighborhoods_population.csv` | `Population_NonWhite` |
| Distance from school to neighborhood | \(d_{sn}\) | `distance.csv` | `N01`–`N31` |
| District white ratio | \(r\) | constant | 0.60 |
| Tolerance | \(\tau\) | constant | 0.10 |

**Decision Variables**  

\[
\begin{aligned}
x_{sn}^{W} &\in \mathbb{Z}_{+} &&\text{white pupils from }n\text{ assigned to }s,\\
x_{sn}^{NW}&\in \mathbb{Z}_{+} &&\text{non‑white pupils from }n\text{ assigned to }s.
\end{aligned}
\]

**Objective** – minimize total travel distance

\[
\min \;\sum_{s\in S}\sum_{n\in N} d_{sn}\,\bigl(x_{sn}^{W}+x_{sn}^{NW}\bigr).
\]

**Constraints**

1. **Neighborhood population balance**

\[
\sum_{s\in S}x_{sn}^{W}=W_n,\qquad
\sum_{s\in S}x_{sn}^{NW}=NW_n,\qquad \forall n\in N.
\]

2. **School capacity**

\[
\sum_{n\in N}\bigl(x_{sn}^{W}+x_{sn}^{NW}\bigr)\le C_s,\qquad \forall s\in S.
\]

3. **Racial balance at each school**

Let \(T_s=\sum_{n\in N}\bigl(x_{sn}^{W}+x_{sn}^{NW}\bigr)\) and \(W_s=\sum_{n\in N}x_{sn}^{W}\).

\[
0.5\,T_s \;\le\; W_s \;\le\; 0.7\,T_s,\qquad \forall s\in S.
\]

(These inequalities enforce \(|W_s/T_s-0.60|\le0.10\).)

4. **Non‑negativity and integrality**

\[
x_{sn}^{W},\,x_{sn}^{NW}\in\mathbb{Z}_{+},\qquad \forall s\in S,\;n\in N.
\]

---

## Data Mapping

| Table ID | Parameter | Column(s) | Mapping |
|----------|-----------|-----------|---------|
| `file_0_view_0` | \(C_s\) | `Capacity` | `School` → \(s\) |
| `file_1_view_0` | \(W_n\) | `Population_White` | `Neighborhood` → \(n\) |
| `file_1_view_0` | \(NW_n\) | `Population_NonWhite` | `Neighborhood` → \(n\) |
| `file_2_view_0` | \(d_{sn}\) | `N01`–`N31` | `School` → \(s\), `Nxx` → \(n\) |

All other constants (\(r=0.60\), \(\tau=0.10\)) are fixed in the model. No additional data are required.