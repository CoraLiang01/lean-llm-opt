**Mathematical Model**

Let  

- \(I=\{S1,S2,\dots ,S12\}\) suppliers  
- \(J=\{C1,C2,\dots ,C12\}\) customers  

Parameters  

- \(f_i\) fixed cost of opening supplier \(i\)  
- \(d_j\) demand of customer \(j\)  
- \(c_{ij}\) transportation cost per unit from supplier \(i\) to customer \(j\)  
- \(M=\sum_{j\in J} d_j\) big‑\(M\) upper bound on shipments from any supplier  

Decision variables  

- \(y_i\in\{0,1\}\) 1 if supplier \(i\) is opened, 0 otherwise  
- \(x_{ij}\ge 0\) quantity shipped from supplier \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \quad\text{(demand)}\\[2pt]
&\sum_{j\in J} x_{ij}\le M\,y_i , &&\forall i\in I \quad\text{(activation)}\\[2pt]
&x_{ij}\ge 0 , &&\forall i\in I,\;j\in J\\
&y_i\in\{0,1\} , &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source file | Source columns | Mapping rule |
|-----------|-------------|----------------|--------------|
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv` | `customer`, `demand` | \(d_j =\) value in column `demand` where `customer` = \(j\) |
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv` | `Unnamed: 0`, `fixed_costs` | \(f_i =\) value in column `fixed_costs` where `Unnamed: 0` = \(i\) |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv` | `Unnamed: 0`, `C1`–`C12` | \(c_{ij} =\) value in column `Cj` where `Unnamed: 0` = \(i\) |
| \(M\) | computed | – | \(M = \sum_{j\in J} d_j\) |

All identifiers (`S1`–`S12`, `C1`–`C12`) are taken verbatim from the CSV files. No additional constraints or capacities are imposed beyond those listed.