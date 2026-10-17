**Mathematical Model**

Let  

* \(S\) – set of Walmart stores (indices from *file_1_view_0*).  
* \(C\) – set of customer groups (indices from *file_0_view_0*).  

Parameters  

* \(d_c\) – daily demand of customer \(c\) (from *file_0_view_0*).  
* \(\text{cap}_s\) – daily supply capacity of store \(s\) (from *file_1_view_0*).  
* \(c_{s,c}\) – transportation cost per unit from store \(s\) to customer \(c\) (from *file_2_view_0*).  

Decision variables  

* \(x_{s,c}\ge 0\) – quantity shipped from store \(s\) to customer \(c\).  

\[
\begin{aligned}
\min\;& \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c} \\[4pt]
\text{s.t.}\;& \sum_{s\in S} x_{s,c}= d_c , &&\forall c\in C \quad\text{(demand satisfaction)}\\[4pt]
& \sum_{c\in C} x_{s,c}\le \text{cap}_s , &&\forall s\in S \quad\text{(capacity)}\\[4pt]
& x_{s,c}\ge 0 , &&\forall s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(C\) | *file_0_view_0* | `customer` | Each distinct value of `customer` forms an index in \(C\). |
| \(d_c\) | *file_0_view_0* | `demand` | For each row, `demand` value is assigned to the corresponding \(c\in C\). |
| \(S\) | *file_1_view_0* | `Unnamed: 0` | Each distinct value of `Unnamed: 0` forms an index in \(S\). |
| \(\text{cap}_s\) | *file_1_view_0* | `supply_capacity` | For each row, `supply_capacity` value is assigned to the corresponding \(s\in S\). |
| \(c_{s,c}\) | *file_2_view_0* | `Unnamed: 0`, `C1`–`C12` | For each row, the value in column `Unnamed: 0` identifies \(s\); each column `Ck` (k=1..12) provides the cost to customer \(Ck\). |

All indices and parameters are taken directly from the full tables; no rows are omitted unless explicitly filtered by the query. No additional constraints or decision variables are introduced beyond those required by the problem statement.