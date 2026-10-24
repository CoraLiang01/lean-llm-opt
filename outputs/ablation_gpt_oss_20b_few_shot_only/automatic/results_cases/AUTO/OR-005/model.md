**Mathematical Model**

Let  

* \(I=\{\,\text{supplier1},\dots ,\text{supplier8}\,\}\) – set of distribution centers.  
* \(J=\{\,\text{demand1},\dots ,\text{demand8}\,\}\) – set of customer groups.  

Parameters  

* \(d_j\) – daily demand of customer \(j\).  
* \(s_i\) – daily supply capacity of supplier \(i\).  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\).  

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\) (continuous).  

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}\;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J .
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source File | Source Column(s) | Notes |
|-----------|-------------|------------------|-------|
| \(d_j\) | `Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv` | `demand1` … `demand8` | Each row corresponds to a customer; column header is the customer ID. |
| \(s_i\) | `Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv` | `supplier1` … `supplier8` | Each row corresponds to a supplier; column header is the supplier ID. |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv` | First column `Unnamed: 0` (supplier IDs) and columns `demand1` … `demand8` (customer IDs) | Each cell \(c_{ij}\) is the cost value at row supplier \(i\), column customer \(j\). |

All identifiers (`supplier1`–`supplier8`, `demand1`–`demand8`) are taken verbatim from the CSV headers/rows. No additional data or aggregation is introduced.