**Mathematical Model**

Let  

- \(I=\{S1,S2,S3,S4,S5,S6\}\) (suppliers)  
- \(J=\{C1,C2,C3,C4,C5,C6\}\) (stores)

Parameters  

- \(f_i\) fixed cost of opening supplier \(i\)  
- \(d_j\) demand of store \(j\)  
- \(c_{ij}\) unit transportation cost from supplier \(i\) to store \(j\)  
- \(M=\sum_{j\in J} d_j\) big‑\(M\) upper bound on shipments from any supplier  

Decision variables  

- \(y_i\in\{0,1\}\) 1 if supplier \(i\) is opened, 0 otherwise  
- \(x_{ij}\ge 0\) quantity shipped from supplier \(i\) to store \(j\)

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}=d_j , && \forall j\in J \quad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij}\le M\,y_i , && \forall i\in I \quad\text{(activation)}\\[4pt]
& y_i\in\{0,1\}, && \forall i\in I\\
& x_{ij}\ge 0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Mapping |
|-----------|--------------|---------------|---------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | row index `Unnamed: 0` → supplier \(i\) |
| \(d_j\) | `file_0_view_0` | `demand` | row index `customer` → store \(j\) |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C6` | row index `Unnamed: 0` → supplier \(i\); column name `Cj` → store \(j\) |
| \(M\) | computed | – | \(M=\sum_{j\in J} d_j\) |

All numeric values are taken directly from the corresponding CSV cells; no further processing or enumeration is required.