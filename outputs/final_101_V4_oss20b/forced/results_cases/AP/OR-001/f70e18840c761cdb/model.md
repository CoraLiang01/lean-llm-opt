**Mathematical Model**

Let  

* \(S=\{s_1,\dots ,s_{18}\}\) – distribution centers (rows of *supply_capacity.csv*).  
* \(C=\{c_1,\dots ,c_{18}\}\) – customer groups (rows of *customer_demand.csv*).  

Parameters  

* \(d_c\) – daily demand of customer \(c\) (column *demand*).  
* \(u_s\) – daily supply capacity of center \(s\) (column *supply_capacity*).  
* \(c_{sc}\) – transportation cost per unit from center \(s\) to customer \(c\) (matrix *transportation_costs.csv*).  

Decision variable  

* \(x_{sc}\ge 0\) – quantity shipped from \(s\) to \(c\).

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} c_{sc}\,x_{sc} \\[4pt]
\text{s.t.}\;&\sum_{s\in S} x_{sc}= d_c \qquad \forall\,c\in C \quad\text{(demand satisfied)}\\[4pt]
&\sum_{c\in C} x_{sc}\le u_s \qquad \forall\,s\in S \quad\text{(capacity not exceeded)}\\[4pt]
&x_{sc}\ge 0 \qquad \forall\,s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key | Value Column |
|-----------|--------------|-----|--------------|
| \(d_c\) | `file_0_view_0` | `customer = c` | `demand` |
| \(u_s\) | `file_1_view_0` | `Unnamed: 0 = s` | `supply_capacity` |
| \(c_{sc}\) | `file_2_view_0` | `Unnamed: 0 = s`, column `C{c}` | cost value |

*Index sets*  
- \(S = \{s \mid s \text{ appears in column } \text{Unnamed: 0} \text{ of } \text{file\_1\_view\_0}\}\)  
- \(C = \{c \mid c \text{ appears in column } \text{customer} \text{ of } \text{file\_0\_view\_0}\}\)

No numerical values are enumerated; the model references the data via the mapping above.