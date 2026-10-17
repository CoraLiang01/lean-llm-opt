**Mathematical Model**

Let  

- \(S=\{S1,\dots ,S10\}\) suppliers  
- \(C=\{C1,\dots ,C10\}\) customers  

Parameters  

- \(cap_s\) daily supply capacity of supplier \(s\)  
- \(dem_c\) daily demand of customer \(c\)  
- \(c_{sc}\) transportation cost per unit from supplier \(s\) to customer \(c\)  

Decision variables  

- \(x_{sc}\in \mathbb{Z}_{\ge 0}\) units shipped from \(s\) to \(c\)

\[
\begin{aligned}
\min\;& \sum_{s\in S}\sum_{c\in C} c_{sc}\,x_{sc} \\[4pt]
\text{s.t.}\;& \sum_{c\in C} x_{sc} \;\le\; cap_s , && \forall s\in S \\[4pt]
& \sum_{s\in S} x_{sc} \;=\; dem_c , && \forall c\in C \\[4pt]
& x_{sc}\;\in\;\mathbb{Z}_{\ge 0}, && \forall s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | ID Column | Value Column |
|-----------|----------|-----------|--------------|
| \(cap_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(dem_c\) | `file_0_view_0` | `customer` | `demand` |
| \(c_{sc}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C10` (column per customer) |

All indices \(s\) and \(c\) are taken from the respective ID columns of the tables.