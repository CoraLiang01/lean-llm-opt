**Mathematical Model**

Let  

* \(P=\{p\}\) be the set of production plants (rows of *supply_capacity.csv*).  
* \(C=\{c\}\) be the set of retail customers (rows of *customer_demand.csv*).  

Parameters  

* \(d_c\) – daily demand of customer \(c\) (column **demand** of *customer_demand.csv*).  
* \(s_p\) – production capacity of plant \(p\) (column **supply_capacity** of *supply_capacity.csv*).  
* \(c_{p,c}\) – transportation cost per unit from plant \(p\) to customer \(c\) (matrix *transportation_costs.csv*).  

Decision variables  

* \(x_{p,c}\) – quantity shipped from plant \(p\) to customer \(c\) (continuous, \(x_{p,c}\ge 0\)).  

\[
\begin{aligned}
\min\;& \sum_{p\in P}\sum_{c\in C} c_{p,c}\,x_{p,c} \\[4pt]
\text{s.t.}\;& \sum_{c\in C} x_{p,c}\;\le\; s_p , &&\forall p\in P \quad\text{(supply limits)}\\[4pt]
& \sum_{p\in P} x_{p,c}\;\ge\; d_c , &&\forall c\in C \quad\text{(demand satisfaction)}\\[4pt]
& x_{p,c}\;\ge\;0 , &&\forall p\in P,\;c\in C .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Mapping |
|-----------|----------|-----------|---------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | \(d_c =\) value in column **demand** where **customer** = \(c\) |
| \(s_p\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | \(s_p =\) value in column **supply_capacity** where **Unnamed: 0** = \(p\) |
| \(c_{p,c}\) | `file_2_view_0` | `Unnamed: 0`, `C1`–`C4` | \(c_{p,c} =\) value in column **C\(c\)** where **Unnamed: 0** = \(p\) |

*Index sets*  
- \(P =\) set of values in column **Unnamed: 0** of `file_1_view_0`.  
- \(C =\) set of values in column **customer** of `file_0_view_0`.