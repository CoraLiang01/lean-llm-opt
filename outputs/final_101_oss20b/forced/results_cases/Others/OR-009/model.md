**Mathematical Model**

Let  

* \(S=\{S1,S2,S3,S4\}\) – set of production plants (index \(s\)).  
* \(C=\{C1,C2,C3,C4\}\) – set of retail customers (index \(c\)).  

Decision variables  

\[
x_{s,c}\;\;(\text{units of beverage shipped from plant }s\text{ to customer }c),\qquad x_{s,c}\ge 0.
\]

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{daily demand of customer }c,\\
q_s &\;=\;\text{daily supply capacity of plant }s,\\
c_{s,c} &\;=\;\text{transportation cost per unit from plant }s\text{ to customer }c.
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}.
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{s\in S} x_{s,c}=d_c,\qquad \forall\,c\in C.
   \]

2. **Supply capacity**  
   \[
   \sum_{c\in C} x_{s,c}\le q_s,\qquad \forall\,s\in S.
   \]

3. **Non‑negativity**  
   \[
   x_{s,c}\ge 0,\qquad \forall\,s\in S,\;c\in C.
   \]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | \(d_c = \text{value in column `demand` where `customer`=c}\) |
| \(q_s\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | \(q_s = \text{value in column `supply_capacity` where `Unnamed: 0`=s}\) |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0`, `C1`–`C4` | \(c_{s,c} = \text{value in column `C{c}` where `Unnamed: 0`=s}\) |

All indices \(s\) and \(c\) are taken from the full set of rows returned in the respective tables; no additional filtering is applied. The model uses continuous shipment quantities, consistent with the problem statement.