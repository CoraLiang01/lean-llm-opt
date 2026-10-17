**Mathematical Model**

Let  

* \(S\) – set of distribution centers (rows of *supply_capacity.csv*).  
* \(C\) – set of customer groups (rows of *customer_demand.csv*).  

Decision variables  

\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

Parameters  

\[
d_c \;(\text{demand of customer }c),\qquad 
s_s \;(\text{supply capacity of center }s),\qquad 
c_{s,c}\;(\text{transport cost per unit from }s\text{ to }c)
\]

---

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c} \\[4pt]
\text{s.t.}\;&\sum_{s\in S} x_{s,c}= d_c \qquad \forall c\in C \\[4pt]
&\sum_{c\in C} x_{s,c}\le s_s \qquad \forall s\in S \\[4pt]
&x_{s,c}\in\mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column(s) | Value Column |
|-----------|--------------|---------------|--------------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` |
| \(s_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (row id), `customer` (column id) | cost value |

*Index sets*  
- \(S\) is the set of all distinct values in `file_1_view_0.Unnamed: 0`.  
- \(C\) is the set of all distinct values in `file_0_view_0.customer`.  

All mappings use exact column names as they appear in the CSV files.