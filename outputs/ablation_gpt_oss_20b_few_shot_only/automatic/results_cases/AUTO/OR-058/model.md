**Mathematical Model**

Let  

- \(I=\{S1,S2,S3,S4,S5,S6\}\) (suppliers)  
- \(J=\{C1,C2,C3,C4,C5,C6\}\) (stores)

Parameters  

- \(f_i\) fixed cost of opening supplier \(i\) \((i\in I)\)  
- \(d_j\) demand of store \(j\) \((j\in J)\)  
- \(c_{ij}\) transportation cost per unit from supplier \(i\) to store \(j\) \((i\in I,\;j\in J)\)  
- \(M=\sum_{j\in J} d_j\) big‑\(M\) upper bound on total shipment from any supplier

Decision variables  

- \(y_i\in\{0,1\}\) 1 if supplier \(i\) is opened, 0 otherwise  
- \(x_{ij}\ge 0\) quantity shipped from supplier \(i\) to store \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. }&\sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\le M\,y_i , &&\forall i\in I \quad\text{(activation)}\\[4pt]
&x_{ij}\ge 0 , &&\forall i\in I,\;j\in J\\
&y_i\in\{0,1\} , &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Source Column | Notes |
|-----------|-------------|---------------|-------|
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv` | `fixed_costs` | Row index is supplier ID |
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv` | `customer demand` | Row index is store ID |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv` | columns `C1`–`C6` | Row index is supplier ID |
| \(M\) | computed | – | \(M=\sum_{j\in J} d_j\) |

All indices, domains, and constraints are defined solely from the query and the supplied CSV files. No additional limits or capacities are imposed.