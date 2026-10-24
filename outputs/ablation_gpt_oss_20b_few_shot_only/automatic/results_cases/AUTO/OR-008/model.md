**Mathematical Model**

Let  

- \(I=\{\,\text{Supplier1},\text{Supplier2},\text{Supplier3},\text{Supplier4},\text{Supplier5}\,\}\)  
- \(J=\{\,\text{Customer1},\text{Customer2},\text{Customer3},\text{Customer4},\text{Customer5},\text{Customer6}\,\}\)

Parameters  

- \(s_i\) – supply capacity of supplier \(i\) (\(i\in I\))  
- \(d_j\) – daily demand of customer \(j\) (\(j\in J\))  
- \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\) (\(i\in I,\;j\in J\))

Decision variables  

- \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\) (\(i\in I,\;j\in J\))

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

| Parameter | Source File | Source Column(s) | Mapping Expression |
|-----------|-------------|------------------|---------------------|
| \(s_i\) | `Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv` | `Suppliers` (row identifier), `supply_capacity` | \(s_{\text{Supplier}k} =\) value in column `supply_capacity` where row label = `Supplierk` |
| \(d_j\) | `Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv` | `Customer1`–`Customer6` (columns) | \(d_{\text{Customer}k} =\) value in column `Customerk` |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv` | `Unnamed: 0` (row identifier), `Customer1`–`Customer6` (columns) | \(c_{\text{Supplier}p,\text{Customer}q} =\) value in cell at row where `Unnamed: 0`=`Supplierp` and column `Customerq` |

All identifiers (`Supplier1`–`Supplier5`, `Customer1`–`Customer6`) are taken verbatim from the respective files. No additional data or aggregation is introduced.