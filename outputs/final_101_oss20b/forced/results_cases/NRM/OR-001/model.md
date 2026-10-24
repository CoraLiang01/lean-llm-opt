**Mathematical Model**

Let  

* \(S=\{S_1,\dots ,S_{18}\}\) – set of distribution centers (suppliers)  
* \(C=\{C_1,\dots ,C_{18}\}\) – set of customer groups  

Parameters  

* \(d_c\) – daily demand of customer \(c\)  (\(c\in C\))  
* \(s_s\) – daily supply capacity of supplier \(s\)  (\(s\in S\))  
* \(c_{s,c}\) – transportation cost per unit from supplier \(s\) to customer \(c\)  (\(s\in S,\;c\in C\))  

Decision variables  

* \(x_{s,c}\in \mathbb{Z}_{\ge 0}\) – quantity shipped from \(s\) to \(c\).

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c} \\[4pt]
\text{s.t. } &\sum_{c\in C} x_{s,c}\;\le\; s_s , &&\forall s\in S \quad\text{(supply capacity)}\\[4pt]
&\sum_{s\in S} x_{s,c}\;=\; d_c , &&\forall c\in C \quad\text{(demand satisfaction)}\\[4pt]
&x_{s,c}\;\in\;\mathbb{Z}_{\ge 0}, &&\forall s\in S,\;c\in C .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(d_c\) | `file_0_view_0` | `demand` | Row identifier: `customer` (values `C1`–`C18`) |
| \(s_s\) | `file_1_view_0` | `supply_capacity` | Row identifier: `Unnamed: 0` (values `S1`–`S18`) |
| \(c_{s,c}\) | `file_2_view_0` | `C1`–`C18` | Row identifier: `Unnamed: 0` (values `S1`–`S18`); column identifier: `C1`–`C18` (customers) |

All parameters are taken directly from the corresponding CSV files; no additional data or transformations are introduced.