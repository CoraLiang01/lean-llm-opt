**Mathematical model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}\;\ge\; d_j \qquad \forall j\in J \\[4pt]
&x_{ij}\;\le\; M\,y_i \qquad \forall i\in I,\; j\in J \\[4pt]
&x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J \\[4pt]
&y_i\in\{0,1\} \qquad \forall i\in I
\end{aligned}
\]

where  

* \(I=\{S1,S2,\dots ,S12\}\) – suppliers,  
* \(J=\{C1,C2,\dots ,C12\}\) – customers,  
* \(M=\sum_{j\in J} d_j\) (any sufficiently large constant, e.g. total demand).

---

**Data‑mapping rules**

| Parameter | Source table | Source column | Mapping description |
|-----------|--------------|---------------|---------------------|
| \(d_j\) | `file_0_view_0` | `demand` | Demand of customer \(j\) |
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Fixed opening cost of supplier \(i\) |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C12` | Transportation cost per unit from supplier \(i\) to customer \(j\) |
| \(I\) | `file_1_view_0` | `Unnamed: 0` | Supplier identifiers |
| \(J\) | `file_0_view_0` | `customer` | Customer identifiers |

All indices and parameters are taken directly from the corresponding source tables; no numerical values are enumerated in the model.