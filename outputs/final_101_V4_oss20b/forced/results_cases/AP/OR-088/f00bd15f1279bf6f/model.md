**Mathematical Model**

Let  

* \(I=\{F1,\dots ,F15\}\) – set of candidate plants  
* \(J=\{C1,\dots ,C15\}\) – set of customers  

Parameters  

* \(f_i\) – fixed opening cost of plant \(i\)  
* \(c_i\) – capacity of plant \(i\)  
* \(t_{ij}\) – per‑unit transport cost from plant \(i\) to customer \(j\)  
* \(d_j\) – demand of customer \(j\)

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if plant \(i\) is opened, 0 otherwise  
* \(x_{ij} \ge 0\) – units shipped from plant \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} = d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij} \le c_i\,y_i \qquad &&\forall i\in I \quad\text{(capacity)}\\[4pt]
& x_{ij}\ge 0 \qquad &&\forall i\in I,\; j\in J\\
& y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(f_i\) | `file_0_view_0` | `fixed_cost` | `f[plant] = fixed_cost` |
| \(c_i\) | `file_0_view_0` | `capacity` | `c[plant] = capacity` |
| \(t_{ij}\) | `file_0_view_0` | `C1`–`C15` | `t[plant,customer] = value of column Cj` |
| \(d_j\) | `file_1_view_0` | `demand` | `d[customer] = demand` |

*Index sets*  
- `I` is the set of values in the `plant` column of `file_0_view_0`.  
- `J` is the set of values in the `customer` column of `file_1_view_0`.  

All parameter values are taken directly from the corresponding CSV records; no numerical values are enumerated in the model.