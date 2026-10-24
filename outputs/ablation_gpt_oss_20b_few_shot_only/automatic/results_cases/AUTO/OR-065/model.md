**Mathematical Model**

Let  

* \(I=\{S_1,S_2,S_3\}\) (warehouses)  
* \(J=\{C_1,C_2,C_3\}\) (customers)

Parameters  

* \(f_i\) fixed cost of opening warehouse \(i\) \((i\in I)\)  
* \(c_{ij}\) transportation cost per unit from warehouse \(i\) to customer \(j\) \((i\in I,\;j\in J)\)  
* \(d_j\) demand of customer \(j\) \((j\in J)\)  
* \(M=\sum_{j\in J} d_j\) big‑\(M\) upper bound on shipments from any warehouse

Decision variables  

* \(y_i\in\{0,1\}\) 1 if warehouse \(i\) is activated, 0 otherwise  
* \(x_{ij}\ge 0\) quantity shipped from warehouse \(i\) to customer \(j\)

---

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i \\[4pt]
\text{s.t. }&\sum_{i\in I} x_{ij}=d_j, &&\forall j\in J \quad\text{(demand)}\\[2pt]
&\sum_{j\in J} x_{ij}\le M\,y_i, &&\forall i\in I \quad\text{(activation)}\\[2pt]
&x_{ij}\ge 0,\; y_i\in\{0,1\}, &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Key Column | Value Column |
|-----------|-------------|------------|--------------|
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv` | `Unnamed: 0` | `fixed_costs` |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv` | `Unnamed: 0` | `C1`, `C2`, `C3` |
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv` | `customer` | `demand` |

*The index sets \(I\) and \(J\) are taken from the unique values in the key columns of the respective files.*