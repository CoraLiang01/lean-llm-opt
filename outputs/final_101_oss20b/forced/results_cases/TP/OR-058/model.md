**Mathematical model**

Let  

* \(I=\{S1,S2,S3,S4,S5,S6\}\) (suppliers)  
* \(J=\{C1,C2,C3,C4,C5,C6\}\) (customers)

Parameters  

* \(f_i\) fixed cost of opening supplier \(i\) (\(i\in I\))  
* \(d_j\) demand of customer \(j\) (\(j\in J\))  
* \(c_{ij}\) transportation cost per unit from supplier \(i\) to customer \(j\) (\(i\in I,\;j\in J\))

Decision variables  

* \(y_i\in\{0,1\}\) 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) quantity shipped from supplier \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; M\,y_i , &&\forall i\in I,\;j\in J \quad\text{(linking, }M\text{ large)}\\[4pt]
&y_i\in\{0,1\}, &&\forall i\in I\\
&x_{ij}\ge 0, &&\forall i\in I,\;j\in J
\end{aligned}
\]

*(If the linking constraint is not required by the problem statement, it may be omitted.)*

---

**Data mapping**

| Parameter | Source table | Column name |
|-----------|--------------|-------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `demand` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C6` (columns) |

*Index sets*  
* Suppliers \(I\) are the values of the column `Unnamed: 0` in `file_1_view_0`.  
* Customers \(J\) are the column headers `C1`–`C6` in `file_2_view_0` (and also the `customer` column in `file_0_view_0`).  

All parameters are taken directly from the corresponding cells of the tables; no aggregation or transformation is performed beyond the linking constraint (if included).