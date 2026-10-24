**Mathematical Model**

Let  

* \(I=\{S1,S2,S3,S4,S5\}\) – set of suppliers  
* \(J=\{C1,C2,C3,C4,C5\}\) – set of customers (branches)  

Parameters  

* \(d_j\) – demand of customer \(j\)  
* \(f_i\) – fixed opening cost of supplier \(i\)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\)  
* \(M=\sum_{j\in J} d_j\) – a sufficiently large constant  

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}= d_j , && \forall j\in J \quad\text{(demand)}\\[4pt]
& x_{ij}\le M\,y_i , && \forall i\in I,\; j\in J \quad\text{(linking)}\\[4pt]
& y_i\in\{0,1\}, && \forall i\in I\\
& x_{ij}\ge 0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(d_j\) | `file_0_view_0` | `demand` |
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(c_{ij}\) | `file_2_view_0` | `C1`, `C2`, `C3`, `C4`, `C5` (corresponding to customer \(j\)) |
| \(M\) | – | computed as \(\sum_{j} d_j\) |

All indices, parameters, and variables are defined exactly as they appear in the source tables; no additional data are introduced.