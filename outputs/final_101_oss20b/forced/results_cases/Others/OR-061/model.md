**Mathematical Model**

Let  

* \(I=\{S1,S2,S3,S4,S5\}\) – suppliers  
* \(J=\{C1,C2,C3,C4,C5\}\) – customers  

Parameters  

* \(f_i\) – fixed cost of opening supplier \(i\)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\)  
* \(d_j\) – demand of customer \(j\)  

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \\[4pt]
&x_{ij}\;\le\; M\,y_i , &&\forall i\in I,\; j\in J \\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J \\[4pt]
&y_i\in\{0,1\} , &&\forall i\in I
\end{aligned}
\]

\(M\) is a sufficiently large constant (e.g., \(\sum_{j} d_j\)).

---

**Data Mapping**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row id column `Unnamed: 0` gives supplier id |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C5` | Row id column `Unnamed: 0` gives supplier id; column headers give customer id |
| \(d_j\) | `file_0_view_0` | `demand` | Row id column `customer` gives customer id |

Index sets \(I\) and \(J\) are taken from the row and column identifiers of the respective tables. No additional filtering is applied.