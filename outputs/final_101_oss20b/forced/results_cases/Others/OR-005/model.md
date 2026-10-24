## Mathematical Model

Let  

* \(S=\{s_1,\dots ,s_8\}\) – set of distribution centers (suppliers)  
* \(C=\{c_1,\dots ,c_8\}\) – set of customer groups  

Decision variable  
\[
x_{s,c}\;\;(\text{units shipped from supplier }s\text{ to customer }c),\qquad x_{s,c}\ge 0
\]

Parameters  

* \(d_c\) – daily demand of customer \(c\) (from **file_0_view_0**)  
* \(q_s\) – daily supply capacity of supplier \(s\) (from **file_1_view_0**)  
* \(c_{s,c}\) – transportation cost per unit from supplier \(s\) to customer \(c\) (from **file_2_view_0**)

### Objective
\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

### Constraints  

Demand satisfaction  
\[
\sum_{s\in S} x_{s,c}=d_c ,\qquad \forall\,c\in C
\]

Supply capacity  
\[
\sum_{c\in C} x_{s,c}\le q_s ,\qquad \forall\,s\in S
\]

Non‑negativity  
\[
x_{s,c}\ge 0 ,\qquad \forall\,s\in S,\;c\in C
\]

---

## Data Mapping

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(d_c\) | **file_0_view_0** | `Customers`, `demand` | \(d_{c_i}= \text{value of } \text{demand}\) for row \(i\) |
| \(q_s\) | **file_1_view_0** | `Supplier`, `supply_capacity` | \(q_{s_j}= \text{value of } \text{supply_capacity}\) for row \(j\) |
| \(c_{s,c}\) | **file_2_view_0** | `Unnamed: 0` (supplier id), `demand1`–`demand8` | \(c_{s_j,c_i}= \text{value of column } \text{demand}_{i}\) in row \(j\) |

*Index sets*  
\(S=\{1,\dots,8\}\) (rows of **file_1_view_0** and **file_2_view_0**)  
\(C=\{1,\dots,8\}\) (rows of **file_0_view_0** and columns `demand1`–`demand8` of **file_2_view_0**)

All parameters are taken directly from the supplied tables; no additional data or assumptions are introduced.