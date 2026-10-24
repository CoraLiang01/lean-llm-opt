**Mathematical Model**

Let  

- \(W\) – set of warehouses (indexed by the values in column **Unnamed: 0** of *supply_capacity.csv*).  
- \(C\) – set of customers (indexed by the values in column **customer** of *customer_demand.csv*).  

Parameters  

- \(d_c\) – daily demand of customer \(c\) (from *customer_demand.csv*, column **demand**).  
- \(s_w\) – daily supply capacity of warehouse \(w\) (from *supply_capacity.csv*, column **supply_capacity**).  
- \(c_{w,c}\) – unit transportation cost from warehouse \(w\) to customer \(c\) (from *transportation_costs.csv*, matrix with rows **Unnamed: 0** and columns **C1**–**C10**).  

Decision variables  

- \(x_{w,c}\) – quantity shipped from warehouse \(w\) to customer \(c\) (continuous, \(x_{w,c}\ge 0\)).  

\[
\begin{aligned}
\min\;& \sum_{w\in W}\sum_{c\in C} c_{w,c}\,x_{w,c} \\[4pt]
\text{s.t. } & \sum_{w\in W} x_{w,c} = d_c && \forall\,c\in C \quad\text{(demand satisfied)}\\[4pt]
& \sum_{c\in C} x_{w,c} \le s_w && \forall\,w\in W \quad\text{(supply limit)}\\[4pt]
& x_{w,c}\ge 0 && \forall\,w\in W,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Notes |
|-----------|--------------|------------|--------------|-------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` | Customer demand (10 rows) |
| \(s_w\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | Warehouse capacity (10 rows) |
| \(c_{w,c}\) | `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C10` (columns) | 10 × 10 cost matrix |

All indices \(w\) and \(c\) are taken from the respective key columns; no additional filtering or enumeration is applied. The model uses the full data set as provided.