**Mathematical Model**

Let  

* \(I=\{S1,S2,\dots ,S10\}\) – set of suppliers  
* \(J=\{C1,C2,\dots ,C10\}\) – set of customer groups  

Parameters  

* \(s_i\) – daily supply capacity of supplier \(i\)  \(i\in I\)  
* \(d_j\) – daily demand of customer group \(j\)  \(j\in J\)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\) \(i\in I,\;j\in J\)

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\) \(i\in I,\;j\in J\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij} \;\le\; s_i , &&\forall i\in I \quad\text{(supply capacity)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Source Column(s) | Mapping |
|-----------|-------------|------------------|---------|
| \(s_i\) | `Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv` | `Unnamed: 0` (supplier ID), `supply_capacity` | `s_i = supply_capacity` for each row |
| \(d_j\) | `Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv` | `Unnamed: 0` (customer ID), `demand` | `d_j = demand` for each row |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv` | `Unnamed: 0` (supplier ID) and columns `C1`–`C10` (customer IDs) | `c_{ij}` taken from the cell at row supplier \(i\), column customer \(j\) |

All identifiers (`S1`–`S10`, `C1`–`C10`) are taken verbatim from the CSV files; no aggregation or renaming is performed. The model uses the full set of suppliers and customers present in the data.