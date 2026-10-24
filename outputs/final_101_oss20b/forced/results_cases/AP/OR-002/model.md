**Mathematical Model**

Let  

- \(S=\{S_1,\dots ,S_{11}\}\) be the set of Walmart stores.  
- \(C=\{C_1,\dots ,C_{12}\}\) be the set of customer groups.  

Decision variables  

\[
x_{s,c}\;\;(\text{units shipped from store }s\text{ to customer }c),\qquad
x_{s,c}\ge 0.
\]

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{daily demand of customer }c,\\
s_s &\;=\;\text{daily supply capacity of store }s,\\
c_{s,c} &\;=\;\text{transportation cost per unit from store }s\text{ to customer }c.
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}.
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{s\in S} x_{s,c}=d_c \quad \forall\,c\in C.
   \]

2. **Supply capacity**  
   \[
   \sum_{c\in C} x_{s,c}\le s_s \quad \forall\,s\in S.
   \]

3. **Non‑negativity**  
   \[
   x_{s,c}\ge 0 \quad \forall\,s\in S,\;c\in C.
   \]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(d_c\)   | `file_0_view_0` | `customer` | `demand` |
| \(s_s\)   | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (row id) | `C1`–`C12` (columns) |

All index sets \(S\) and \(C\) are taken from the row and column identifiers of the respective source tables. No numerical values are enumerated in the model.