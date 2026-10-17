**Mathematical model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}\;\ge\; d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; y_i\,d_j \qquad &&\forall i\in I,\; j\in J \quad\text{(open‑supplier link)}\\[4pt]
&x_{ij}\;\ge\;0 \qquad &&\forall i\in I,\; j\in J\\
&y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

*Sets*  
\(I=\{S1,S2,\dots ,S12\}\) (suppliers)  
\(J=\{C1,C2,\dots ,C12\}\) (customers)

*Parameters*  
\(d_j\) – demand of customer \(j\) (from **file_0_view_0**, column **demand**)  
\(f_i\) – fixed opening cost of supplier \(i\) (from **file_1_view_0**, column **fixed_costs**)  
\(c_{ij}\) – unit transportation cost from supplier \(i\) to customer \(j\) (from **file_2_view_0**, columns **C1**–**C12**)

*Decision variables*  
\(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
\(x_{ij}\ge0\) – quantity shipped from supplier \(i\) to customer \(j\)

---

**Data‑mapping rules**

| Parameter | Source table | Source column | Mapping description |
|-----------|--------------|---------------|---------------------|
| \(d_j\) | `file_0_view_0` | `demand` | Demand of customer \(j\) |
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Fixed cost of supplier \(i\) |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C12` | Unit transportation cost from supplier \(i\) to customer \(j\) |

All indices \(i\) and \(j\) are taken from the row identifiers (`Unnamed: 0` for suppliers, `customer` for customers) in the respective tables. No additional data are introduced beyond what is supplied in the observation.