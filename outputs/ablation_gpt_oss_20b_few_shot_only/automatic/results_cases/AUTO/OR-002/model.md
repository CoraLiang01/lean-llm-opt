**Mathematical Model**

Let  

* \(I=\{S1,S2,\dots ,S11\}\) – set of Walmart stores  
* \(J=\{C1,C2,\dots ,C12\}\) – set of customer groups  

Parameters  

* \(d_j\) : daily demand of customer \(j\) (from *customer_demand.csv*)  
* \(s_i\) : daily supply capacity of store \(i\) (from *supply_capacity.csv*)  
* \(c_{ij}\) : transportation cost per unit from store \(i\) to customer \(j\) (from *transportation_costs.csv*)

Decision variables  

* \(x_{ij}\ge 0\) : quantity shipped from store \(i\) to customer \(j\) (continuous)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij} \;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source File | Column(s) | Notes |
|-----------|-------------|-----------|-------|
| \(d_j\) | `Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv` | `customer` (row header), `demand` (value) | Each row gives one customer’s demand. |
| \(s_i\) | `Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv` | `Unnamed: 0` (row header), `supply_capacity` (value) | Each row gives one store’s capacity. |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv` | `Unnamed: 0` (row header for stores), columns `C1`–`C12` (values) | Each cell gives cost from store \(i\) to customer \(j\). |

All identifiers (`S1`–`S11`, `C1`–`C12`) are taken verbatim from the source files; no aggregation or renaming is performed. The model uses the full sets of stores and customers present in the data.