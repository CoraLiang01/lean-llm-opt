**Mathematical Model**

Let  

* \(W=\{w\}\) be the set of warehouses (indices taken from the column **Unnamed: 0** in *file_1_view_0*).  
* \(C=\{c\}\) be the set of customers (indices taken from the column **customer** in *file_0_view_0*).  

Parameters  

* \(d_c\) – demand of customer \(c\) (column **demand** in *file_0_view_0*).  
* \(s_w\) – supply capacity of warehouse \(w\) (column **supply_capacity** in *file_1_view_0*).  
* \(c_{wc}\) – transportation cost per unit from warehouse \(w\) to customer \(c\) (columns **C1**–**C10** in *file_2_view_0*).  

Decision variables  

* \(x_{wc}\in \mathbb{Z}_{\ge 0}\) – number of units shipped from warehouse \(w\) to customer \(c\).

\[
\begin{aligned}
\min\;& \sum_{w\in W}\sum_{c\in C} c_{wc}\,x_{wc} \\[4pt]
\text{s.t.}\;& \sum_{w\in W} x_{wc}= d_c , &&\forall c\in C \quad\text{(demand satisfaction)}\\[4pt]
& \sum_{c\in C} x_{wc}\le s_w , &&\forall w\in W \quad\text{(supply capacity)}\\[4pt]
& x_{wc}\in \mathbb{Z}_{\ge 0}, &&\forall w\in W,\;c\in C .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Key Column | Value Column(s) |
|-----------|----------|------------|-----------------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` |
| \(s_w\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{wc}\) | `file_2_view_0` | `Unnamed: 0` | `C1`, `C2`, …, `C10` |

*The index sets \(W\) and \(C\) are derived from the key columns of the corresponding tables.*