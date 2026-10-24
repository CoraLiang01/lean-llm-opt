**Mathematical Model**

Let  

\[
I=\{S1,S2,S3,S4,S5\},\qquad  
J=\{C1,C2,C3,C4,C5\}
\]

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of supplier }i\\
t_{ij}&\;=\;\text{transportation cost per unit from supplier }i\text{ to customer }j\\
d_j &\;=\;\text{demand of customer }j\\
M &= \sum_{j\in J} d_j \;=\;187
\end{aligned}
\]

Decision variables  

\[
y_i \in \{0,1\}\quad\text{(supplier }i\text{ is open)}\\
x_{ij}\;\ge 0\quad\text{(units shipped from }i\text{ to }j)
\]

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij}\;\ge\; d_j \qquad \forall j\in J \\[4pt]
&x_{ij}\;\le\; M\,y_i \qquad \forall i\in I,\; j\in J \\[4pt]
&y_i\in\{0,1\}\qquad \forall i\in I \\[2pt]
&x_{ij}\ge 0\qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

### Data Mapping

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row identified by `Unnamed: 0` = supplier id |
| \(t_{ij}\) | `file_2_view_0` | `C1`–`C5` | Row identified by `Unnamed: 0` = supplier id; column `Cj` = cost to customer \(j\) |
| \(d_j\) | `file_0_view_0` | `demand` | Row identified by `customer` = customer id |
| \(M\) | — | — | Computed as \(\sum_j d_j = 187\) from `file_0_view_0` |

All index sets \(I\) and \(J\) are taken from the full set of suppliers and customers returned in the observation. No additional filtering is applied. The model uses continuous shipment variables and binary opening variables as specified.