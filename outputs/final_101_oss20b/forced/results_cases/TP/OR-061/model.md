**Mathematical model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}\;\ge\; d_j \qquad &&\forall j\in J \\[4pt]
&x_{ij}\;\le\; M\,y_i \qquad &&\forall i\in I,\; j\in J \\[4pt]
&y_i\in\{0,1\} \qquad &&\forall i\in I \\[2pt]
&x_{ij}\;\ge\;0 \qquad &&\forall i\in I,\; j\in J
\end{aligned}
\]

* \(I=\{S1,S2,S3,S4,S5\}\) – suppliers  
* \(J=\{C1,C2,C3,C4,C5\}\) – customer branches  
* \(d_j\) – demand of branch \(j\) (from **file_0_view_0**, column *demand*)  
* \(f_i\) – fixed opening cost of supplier \(i\) (from **file_1_view_0**, column *fixed_costs*)  
* \(c_{ij}\) – unit transportation cost from supplier \(i\) to branch \(j\) (from **file_2_view_0**, columns *C1*–*C5*)  
* \(M=\sum_{j\in J} d_j\) – a sufficiently large constant (e.g., total demand)  

---

**Data‑mapping rules**

| Parameter | Source table | Source column |
|-----------|--------------|---------------|
| \(d_j\)   | `file_0_view_0` | `demand` |
| \(f_i\)   | `file_1_view_0` | `fixed_costs` |
| \(c_{ij}\)| `file_2_view_0` | `C1`–`C5` (column matching customer \(j\)) |

All indices and parameters are taken directly from the returned rows of the respective files; no aggregation or omission is performed. The model uses the binary decision variables \(y_i\) to activate suppliers and continuous variables \(x_{ij}\) to allocate shipments, ensuring all customer demands are met at minimum total cost.