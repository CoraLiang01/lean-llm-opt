**Mathematical Model**

Let  

* \(S\) – set of Walmart stores (indexed by the values in column **`Unnamed: 0`** of *file_1_view_0*).  
* \(C\) – set of customer groups (indexed by the values in column **`customer`** of *file_0_view_0*).  

Parameters  

* \(d_c\) – daily demand of customer \(c\) (from *file_0_view_0*, column **`demand`**).  
* \(s_s\) – daily supply capacity of store \(s\) (from *file_1_view_0*, column **`supply_capacity`**).  
* \(c_{s,c}\) – transportation cost per unit from store \(s\) to customer \(c\) (from *file_2_view_0*, row identified by **`Unnamed: 0`** = \(s\) and column \(c\)).  

Decision variables  

\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\quad\forall s\in S,\;c\in C
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

Supply‑capacity constraints  

\[
\sum_{c\in C} x_{s,c}\;\le\; s_s \qquad \forall s\in S
\]

Demand‑satisfaction constraints  

\[
\sum_{s\in S} x_{s,c}\;=\; d_c \qquad \forall c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column(s) | Value Column |
|-----------|--------------|---------------|--------------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` |
| \(s_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (row key) | column named `C1` … `C12` (column key `customer`) |

*Index sets*  

* \(S\) is the set of distinct values in `file_1_view_0` column `Unnamed: 0`.  
* \(C\) is the set of distinct values in `file_0_view_0` column `customer`.  

All parameters are taken directly from the supplied CSV files; no values are enumerated or altered.