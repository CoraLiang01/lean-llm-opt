## Mathematical Model

Let  

- \(W=\{S1,\dots ,S10\}\) : set of warehouses  
- \(C=\{C1,\dots ,C10\}\) : set of customers  

Decision variables  

\[
x_{w,c}\;\;(\text{units shipped from warehouse }w\text{ to customer }c),\qquad
x_{w,c}\ge 0
\]

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{daily demand of customer }c,\\
s_w &\;=\;\text{daily supply capacity of warehouse }w,\\
c_{w,c} &\;=\;\text{transportation cost per unit from }w\text{ to }c .
\end{aligned}
\]

### Objective  

\[
\min \;\sum_{w\in W}\sum_{c\in C} c_{w,c}\,x_{w,c}
\]

### Constraints  

1. **Demand satisfaction** (each customer’s demand must be met)

\[
\sum_{w\in W} x_{w,c}= d_c ,\qquad \forall\,c\in C
\]

2. **Supply capacity** (no warehouse can ship more than its capacity)

\[
\sum_{c\in C} x_{w,c}\le s_w ,\qquad \forall\,w\in W
\]

3. **Non‑negativity**

\[
x_{w,c}\ge 0 ,\qquad \forall\,w\in W,\;c\in C
\]

---

## Data Mapping

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | \(d_c = \text{value of column `demand` where `customer` = }c\) |
| \(s_w\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | \(s_w = \text{value of column `supply_capacity` where `Unnamed: 0` = }w\) |
| \(c_{w,c}\) | `file_2_view_0` | `Unnamed: 0`, `C1`–`C10` | \(c_{w,c} = \text{value of column `C{c}` where `Unnamed: 0` = }w\) |

All indices \(w\) and \(c\) are taken from the full set of rows returned in the observation; no additional filtering is applied. The model uses the exact numeric values from the source tables as shown above.