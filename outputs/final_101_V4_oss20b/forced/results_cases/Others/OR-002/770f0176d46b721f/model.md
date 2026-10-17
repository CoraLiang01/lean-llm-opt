**Mathematical Model**

Let  

* \(S=\{s\mid s\text{ is a Walmart store}\}\) – index set from **file_1_view_0**  
* \(C=\{c\mid c\text{ is a customer group}\}\) – index set from **file_0_view_0**

Parameters  

* \(d_c\) – daily demand of customer \(c\) (units)  
  *source*: **file_0_view_0**, column **demand**  
* \(s_s\) – daily supply capacity of store \(s\) (units)  
  *source*: **file_1_view_0**, column **supply_capacity**  
* \(c_{s,c}\) – transportation cost per unit from store \(s\) to customer \(c\) (RMB)  
  *source*: **file_2_view_0**, matrix with row‑axis **Unnamed: 0** (stores) and column‑axis **customer** (customers)

Decision variables  

* \(x_{s,c}\ge 0\) – quantity transported from store \(s\) to customer \(c\) (continuous)

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

Constraints  

1. **Demand satisfaction** – for every customer \(c\in C\)

\[
\sum_{s\in S} x_{s,c}=d_c
\]

2. **Supply capacity** – for every store \(s\in S\)

\[
\sum_{c\in C} x_{s,c}\le s_s
\]

3. **Non‑negativity**

\[
x_{s,c}\ge 0 \qquad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Index set | Source table | Key column | Description |
|-----------|--------------|------------|-------------|
| \(S\) | **file_1_view_0** | **Unnamed: 0** | Store identifiers (S1…S11) |
| \(C\) | **file_0_view_0** | **customer** | Customer group identifiers (C1…C12) |

| Parameter | Source table | Column(s) | Mapping rule |
|-----------|--------------|-----------|--------------|
| \(d_c\) | **file_0_view_0** | **demand** | \(d_c =\) value in row where **customer** = \(c\) |
| \(s_s\) | **file_1_view_0** | **supply_capacity** | \(s_s =\) value in row where **Unnamed: 0** = \(s\) |
| \(c_{s,c}\) | **file_2_view_0** | matrix cell (row **Unnamed: 0** = \(s\), column **customer** = \(c\)) | \(c_{s,c} =\) cost value in that cell |

All parameters are real numbers extracted directly from the corresponding CSV files; no additional processing or aggregation is performed. The model is a linear program with continuous decision variables and linear constraints.