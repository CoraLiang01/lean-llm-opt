**Mathematical Model**

Let  

- \(S\) – set of distribution centers (rows of *supply_capacity.csv*).  
- \(C\) – set of customer groups (rows of *customer_demand.csv*).  

Parameters  

- \(d_c\) – demand of customer \(c\) (column **demand** in *customer_demand.csv*).  
- \(s_s\) – supply capacity of center \(s\) (column **supply_capacity** in *supply_capacity.csv*).  
- \(c_{s,c}\) – transportation cost per unit from center \(s\) to customer \(c\) (matrix *transportation_costs.csv*).  

Decision variables  

- \(x_{s,c}\in \mathbb{Z}_{\ge 0}\) – quantity shipped from \(s\) to \(c\).  

\[
\begin{aligned}
\min\;& \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c} \\[4pt]
\text{s.t. } & \sum_{s\in S} x_{s,c}= d_c , && \forall c\in C \quad\text{(demand satisfaction)}\\[4pt]
& \sum_{c\in C} x_{s,c}\le s_s , && \forall s\in S \quad\text{(supply capacity)}\\[4pt]
& x_{s,c}\in \mathbb{Z}_{\ge 0}, && \forall s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(d_c\) | `file_0_view_0` | `demand` |
| \(s_s\) | `file_1_view_0` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `C1` … `C18` (columns indexed by customer) |

Indices  

- \(S\) is the set of values in column **Unnamed: 0** of `file_1_view_0`.  
- \(C\) is the set of values in column **customer** of `file_0_view_0`.  

All parameters are taken directly from the corresponding CSV files; no additional data are introduced.