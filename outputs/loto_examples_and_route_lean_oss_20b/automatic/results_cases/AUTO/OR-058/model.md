**Mathematical Model**

Let  

* \(I=\{S1,S2,S3,S4,S5,S6\}\) – suppliers  
* \(J=\{C1,C2,C3,C4,C5,C6\}\) – customers  

Parameters  

* \(f_i\) – fixed cost of opening supplier \(i\)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\)  
* \(d_j\) – demand of customer \(j\) (units)  

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\)  

Let \(M=\sum_{j\in J} d_j\) (a sufficiently large constant).

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}=d_j , && \forall j\in J \quad\text{(demand satisfaction)}\\[4pt]
& x_{ij}\le M\,y_i , && \forall i\in I,\; j\in J \quad\text{(supplier activation)}\\[4pt]
& y_i\in\{0,1\}, && \forall i\in I\\
& x_{ij}\ge 0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

---

### Data Mapping

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | `f_{S1}=98.88`, `f_{S2}=99.73`, …, `f_{S6}=112.65` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C6` | `c_{S1,C1}=0.08`, `c_{S1,C2}=52.33`, …, `c_{S6,C6}=1543.49` |
| \(d_j\) | `file_0_view_0` | `demand` | `d_{C1}=216`, `d_{C2}=216`, …, `d_{C6}=144` |

All indices \(i\) and \(j\) are taken from the full set of rows returned in the observation; no additional filtering is applied. The big‑\(M\) constant is derived from the sum of all customer demands and is not stored in the data tables.