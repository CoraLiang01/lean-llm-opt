**Mathematical Model**

Let  

* \(I=\{S1,S2,\dots ,S10\}\) (warehouses)  
* \(J=\{C1,C2,\dots ,C10\}\) (retail stores)

Parameters  

* \(d_j\) – daily demand of store \(j\) (\(j\in J\))  
* \(s_i\) – daily supply capacity of warehouse \(i\) (\(i\in I\))  
* \(c_{ij}\) – transportation cost per unit from warehouse \(i\) to store \(j\) (\(i\in I,\;j\in J\))

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from warehouse \(i\) to store \(j\) (\(i\in I,\;j\in J\))

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}\;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\;j\in J .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Source Column | Mapping Rule |
|-----------|-------------|---------------|--------------|
| \(d_j\) | `Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv` | `C1`–`C10` | For each row, the value in column `Ck` (k=1..10) is assigned to \(d_{Ck}\). |
| \(s_i\) | `Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv` | `supply_capacity` | For each row, the value in column `supply_capacity` is assigned to \(s_{Si}\). |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv` | `C1`–`C10` | For each row, the value in column `Ck` (k=1..10) is assigned to \(c_{Si,Ck}\). |

All identifiers (`S1`–`S10`, `C1`–`C10`) are taken verbatim from the respective files; no aggregation or renaming is performed. The model uses the full set of entities present in the data.