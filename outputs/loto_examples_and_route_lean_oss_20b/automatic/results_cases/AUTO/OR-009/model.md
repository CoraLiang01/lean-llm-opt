**Symbolic Model**

Let  

* \(S=\{s\}\) be the set of production plants (indices taken from *file_1_view_0* column **Unnamed: 0**).  
* \(C=\{c\}\) be the set of retail customers (indices taken from *file_0_view_0* column **customer**).  

Parameters  

* \(d_c\) – daily demand of customer \(c\) (from *file_0_view_0*, column **demand**).  
* \(s_s\) – production capacity of plant \(s\) (from *file_1_view_0*, column **supply_capacity**).  
* \(c_{s,c}\) – transportation cost per unit from plant \(s\) to customer \(c\) (from *file_2_view_0*, columns **C1**–**C4**).  

Decision variables  

* \(x_{s,c}\ge 0\) – quantity shipped from plant \(s\) to customer \(c\) (continuous).

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c} \\[4pt]
\text{s.t. } &\sum_{s\in S} x_{s,c}= d_c , &&\forall\,c\in C \quad\text{(demand satisfaction)}\\[4pt]
&\sum_{c\in C} x_{s,c}\le s_s , &&\forall\,s\in S \quad\text{(capacity limit)}\\[4pt]
&x_{s,c}\ge 0 , &&\forall\,s\in S,\;c\in C .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| \(d_c\) | *file_0_view_0* | **demand** |
| \(s_s\) | *file_1_view_0* | **supply_capacity** |
| \(c_{s,c}\) | *file_2_view_0* | **C1**, **C2**, **C3**, **C4** (row identified by **Unnamed: 0**) |
| Index set \(S\) | *file_1_view_0* | **Unnamed: 0** |
| Index set \(C\) | *file_0_view_0* | **customer** |

All parameters are mapped directly to the corresponding columns in the supplied CSV views; no additional data are introduced.