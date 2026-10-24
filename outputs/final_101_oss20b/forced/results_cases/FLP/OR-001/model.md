**Mathematical Model**

Let  

* \(S\) – set of suppliers (rows of *supply_capacity.csv*).  
* \(C\) – set of customers (rows of *customer_demand.csv*).  

Parameters  

* \(d_c\) – daily demand of customer \(c\) (column **demand** of *file_0_view_0*).  
* \(\bar{s}_s\) – supply capacity of supplier \(s\) (column **supply_capacity** of *file_1_view_0*).  
* \(c_{s,c}\) – transportation cost per unit from supplier \(s\) to customer \(c\) (matrix in *file_2_view_0*; row key **Unnamed: 0**, column key \(c\)).  

Decision variables  

* \(x_{s,c}\ge 0\) – quantity shipped from supplier \(s\) to customer \(c\).  

\[
\begin{aligned}
\min\;& \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c} \\[4pt]
\text{s.t.}\;& \sum_{s\in S} x_{s,c}= d_c , &&\forall c\in C \quad\text{(demand)}\\[4pt]
& \sum_{c\in C} x_{s,c}\le \bar{s}_s , &&\forall s\in S \quad\text{(capacity)}\\[4pt]
& x_{s,c}\ge 0 , &&\forall s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` |
| \(\bar{s}_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C18` (each customer column) |

*Index sets*  
- \(S =\) distinct values of **Unnamed: 0** in `file_1_view_0` (and `file_2_view_0`).  
- \(C =\) distinct values of **customer** in `file_0_view_0` (and columns `C1`–`C18` in `file_2_view_0`).