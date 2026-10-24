**Mathematical model**

Let  

* \(I=\{S1,S2\}\) – set of suppliers  
* \(J=\{C1,C2\}\) – set of supermarkets  

Parameters  

* \(f_i\) – fixed activation cost of supplier \(i\)  
* \(d_j\) – demand of supermarket \(j\)  
* \(c_{ij}\) – unit transportation cost from supplier \(i\) to supermarket \(j\)

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is activated, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity supplied from \(i\) to \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. }&\sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \quad\text{(meet demand)}\\[2pt]
&x_{ij}\le d_j\,y_i , &&\forall i\in I,\;j\in J \quad\text{(no supply if supplier inactive)}\\[2pt]
&y_i\in\{0,1\}, &&\forall i\in I\\
&x_{ij}\ge 0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Source column | Notes |
|-----------|--------------|---------------|-------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row key `Unnamed: 0` gives supplier id |
| \(d_j\) | `file_0_view_0` | `demand` | Row key `customer` gives supermarket id |
| \(c_{ij}\) | `file_2_view_0` | `C1`, `C2` | Row key `Unnamed: 0` gives supplier id; column names give supermarket id |

All index sets \(I\) and \(J\) are taken from the distinct supplier and customer identifiers present in the respective source tables. No additional data or constraints are introduced beyond those specified in the query.