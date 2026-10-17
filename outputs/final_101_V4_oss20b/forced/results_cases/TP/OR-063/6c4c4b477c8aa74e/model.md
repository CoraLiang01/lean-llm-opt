**Mathematical Model**

Let  

* \(I=\{S_1,S_2,\dots ,S_7\}\) (warehouses)  
* \(J=\{C_1,C_2,\dots ,C_7\}\) (customers)

Parameters  

* \(f_i\) fixed cost of opening warehouse \(i\) \((i\in I)\)  
* \(d_j\) demand of customer \(j\) \((j\in J)\)  
* \(c_{ij}\) transportation cost per unit from warehouse \(i\) to customer \(j\) \((i\in I,\;j\in J)\)  
* \(M_j=d_j\) big‑\(M\) bound for customer \(j\)

Decision variables  

* \(y_i\in\{0,1\}\) 1 if warehouse \(i\) is activated, 0 otherwise  
* \(x_{ij}\ge 0\) quantity shipped from warehouse \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \qquad\text{(demand)}\\[2pt]
&x_{ij} \;\le\; M_j\,y_i , &&\forall i\in I,\;j\in J \qquad\text{(activation)}\\[2pt]
&y_i \in \{0,1\}, &&\forall i\in I\\
&x_{ij}\ge 0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row identifier `Unnamed: 0` gives \(i\) |
| \(d_j\) | `file_0_view_0` | `demand` | Row identifier `customer` gives \(j\) |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C7` | Row identifier `Unnamed: 0` gives \(i\); column header gives \(j\) |
| \(M_j\) | derived | \(d_j\) | Big‑\(M\) bound set equal to demand |

All index sets \(I\) and \(J\) are taken directly from the row identifiers of the respective tables. No additional data or constraints are introduced beyond those specified in the query.