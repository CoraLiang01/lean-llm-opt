**Mathematical Model**

Let  

* \(W\) – set of warehouses (indexed by the values in column **Unnamed: 0** of *file_1_view_0*).  
* \(C\) – set of customers (indexed by the values in column **customer** of *file_0_view_0*).  

Parameters  

* \(d_c\) – daily demand of customer \(c\) (from *file_0_view_0*, column **demand**).  
* \(s_w\) – daily supply capacity of warehouse \(w\) (from *file_1_view_0*, column **supply_capacity**).  
* \(c_{w,c}\) – unit transportation cost from warehouse \(w\) to customer \(c\) (from *file_2_view_0*, matrix with row key **Unnamed: 0** and column keys **C1**–**C10**).  

Decision variables  

* \(x_{w,c}\) – quantity shipped from warehouse \(w\) to customer \(c\) (continuous, \(x_{w,c}\ge 0\)).  

\[
\begin{aligned}
\min\;& \sum_{w\in W}\sum_{c\in C} c_{w,c}\,x_{w,c} \\[4pt]
\text{s.t. } & \sum_{w\in W} x_{w,c} = d_c \quad &&\forall\,c\in C \quad\text{(demand satisfied)}\\[4pt]
& \sum_{c\in C} x_{w,c} \le s_w \quad &&\forall\,w\in W \quad\text{(supply limit)}\\[4pt]
& x_{w,c}\ge 0 \quad &&\forall\,w\in W,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Notes |
|-----------|--------------|------------|--------------|-------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` | Customer demand |
| \(s_w\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | Warehouse capacity |
| \(c_{w,c}\) | `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C10` (columns) | Unit transportation cost matrix |

All index sets \(W\) and \(C\) are taken from the distinct values of the respective key columns in the source tables. No numerical values are enumerated in the model; they are referenced through the mapping above.