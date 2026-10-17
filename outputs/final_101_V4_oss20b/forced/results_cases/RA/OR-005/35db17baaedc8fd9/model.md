**Mathematical Model**

Let  

* \(S=\{s_1,\dots ,s_8\}\) be the set of suppliers (distribution centers).  
* \(C=\{c_1,\dots ,c_8\}\) be the set of customers (customer groups).  

Parameters  

* \(d_c\) – daily demand of customer \(c\).  
* \(s_s\) – daily supply capacity of supplier \(s\).  
* \(c_{s,c}\) – transportation cost per unit from supplier \(s\) to customer \(c\).  

Decision variables  

* \(x_{s,c}\ge 0\) – quantity shipped from supplier \(s\) to customer \(c\) (continuous).  

\[
\begin{aligned}
\min \quad & \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c} \\[4pt]
\text{s.t.}\quad
& \sum_{s\in S} x_{s,c} \;\ge\; d_c , && \forall\,c\in C \quad\text{(demand fulfillment)}\\[4pt]
& \sum_{c\in C} x_{s,c} \;\le\; s_s , && \forall\,s\in S \quad\text{(supply limit)}\\[4pt]
& x_{s,c}\;\ge\;0 , && \forall\,s\in S,\;c\in C .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| \(d_c\) | `file_0_view_0` | `Customers` (index), `demand` |
| \(s_s\) | `file_1_view_0` | `Supplier` (index), `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (supplier index), columns `demand1`–`demand8` (customer indices) |

*Index sets*  
- \(S\) is the set of distinct values in the `Supplier` column of `file_1_view_0`.  
- \(C\) is the set of distinct values in the `Customers` column of `file_0_view_0`.  

All parameter values are taken directly from the corresponding tables; no additional data are introduced.