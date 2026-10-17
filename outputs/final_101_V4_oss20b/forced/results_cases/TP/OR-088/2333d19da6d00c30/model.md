**Mathematical model**

Let  

* \(I=\{F1,\dots ,F15\}\) – set of candidate plants  
* \(J=\{C1,\dots ,C15\}\) – set of customers  

Parameters  

* \(f_i\) – fixed opening cost of plant \(i\)  
* \(s_i\) – capacity of plant \(i\)  
* \(c_{ij}\) – per‑unit transport cost from plant \(i\) to customer \(j\)  
* \(d_j\) – demand of customer \(j\)

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if plant \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from plant \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} \;\ge\; d_j , && \forall j\in J \qquad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij} \;\le\; s_i\,y_i , && \forall i\in I \qquad\text{(capacity)}\\[4pt]
& y_i \in \{0,1\}, && \forall i\in I\\
& x_{ij}\ge 0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data mapping rules**

| Parameter | Source table | Column name | Notes |
|-----------|--------------|-------------|-------|
| \(f_i\) | `file_0_view_0` | `fixed_cost` | Fixed opening cost of plant |
| \(s_i\) | `file_0_view_0` | `capacity` | Plant capacity |
| \(c_{ij}\) | `file_0_view_0` | `C1`–`C15` | Per‑unit transport cost from plant \(i\) to customer \(j\) |
| \(d_j\) | `file_1_view_0` | `demand` | Customer demand |

The index sets \(I\) and \(J\) are obtained from the distinct values of the `plant` and `customer` columns in the respective tables. No additional data are required.