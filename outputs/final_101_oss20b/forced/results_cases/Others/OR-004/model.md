**Mathematical Model**

Let  

* \(S=\{S1,\dots ,S12\}\) – set of distribution centers (suppliers).  
* \(C=\{C1,\dots ,C12\}\) – set of customer groups.  

Decision variable  

\[
x_{s,c}\;\;(\text{units shipped from supplier }s\text{ to customer }c),\qquad
x_{s,c}\ge 0,\; \forall s\in S,\;c\in C .
\]

Parameters  

* \(d_c\) – daily demand of customer \(c\).  
* \(u_s\) – supply capacity of supplier \(s\).  
* \(c_{s,c}\) – transportation cost per unit from \(s\) to \(c\).

Objective (minimise total transportation cost)

\[
\min \;\; \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}.
\]

Constraints  

1. **Demand satisfaction** – each customer’s demand must be met exactly  

\[
\sum_{s\in S} x_{s,c}= d_c ,\qquad \forall c\in C .
\]

2. **Supply capacity** – a supplier cannot ship more than its capacity  

\[
\sum_{c\in C} x_{s,c}\le u_s ,\qquad \forall s\in S .
\]

All variables are continuous and non‑negative.

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | \(d_c =\) value in column `demand` where `customer` = \(c\) |
| \(u_s\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | \(u_s =\) value in column `supply_capacity` where `Unnamed: 0` = \(s\) |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0`, `C1`–`C12` | \(c_{s,c} =\) value in column `C{c}` where `Unnamed: 0` = \(s\) |

Index sets \(S\) and \(C\) are taken from the distinct values of the identifiers in the respective tables. No additional filtering or aggregation is applied. All numerical values are used exactly as provided in the source tables.