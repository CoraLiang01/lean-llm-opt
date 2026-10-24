**Mathematical Model**

Let  

* \(I=\{S1,S2,S3,S4,S5\}\) (warehouses)  
* \(J=\{D1,D2,D3,D4,D5\}\) (stores)

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

**Data Mapping Rules**

| Parameter | Source File | Source Column(s) | Mapping |
|-----------|-------------|------------------|---------|
| \(I\) | `Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv` | Row identifiers (`region`) | Each distinct row identifier becomes an element of \(I\). |
| \(J\) | `Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv` | Row identifiers (`customer`) | Each distinct row identifier becomes an element of \(J\). |
| \(d_j\) | `Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv` | Column `demand` (second column) | For each row \(j\), assign its `demand` value to \(d_j\). |
| \(s_i\) | `Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv` | Column `supply_capacity` (second column) | For each row \(i\), assign its `supply_capacity` value to \(s_i\). |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv` | Rows: warehouse IDs (`Unnamed: 0`); Columns: store IDs (`D1`–`D5`) | For each row \(i\) and column \(j\), assign the cell value to \(c_{ij}\). |

All parameters are non‑negative real numbers. The model uses the exact identifiers and values from the source files; no aggregation or omission is performed.