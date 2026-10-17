**Mathematical Model**

Let  

* \(S\) – set of suppliers (distribution centers)  
* \(C\) – set of customers (customer groups)

Parameters  

\[
\begin{aligned}
d_c &\quad\text{demand of customer }c\in C\\
s_s &\quad\text{supply capacity of supplier }s\in S\\
c_{s,c} &\quad\text{transportation cost per unit from supplier }s\text{ to customer }c
\end{aligned}
\]

Decision variables  

\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

Constraints  

1. **Supply capacity**  
\[
\sum_{c\in C} x_{s,c}\;\le\; s_s \qquad \forall s\in S
\]

2. **Demand satisfaction**  
\[
\sum_{s\in S} x_{s,c}\;=\; d_c \qquad \forall c\in C
\]

3. **Non‑negativity & integrality**  
\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(d_c\) | `file_0_view_0` | `demand` | Row key: `customer` |
| \(s_s\) | `file_1_view_0` | `supply_capacity` | Row key: `Unnamed: 0` (supplier id) |
| \(c_{s,c}\) | `file_2_view_0` | `C1` … `C12` | Row key: `Unnamed: 0` (supplier id); column key: `C1` … `C12` (customer id) |

All indices \(S\) and \(C\) are derived from the distinct values of the row and column keys in the respective tables. No numeric values are embedded in the model; they are supplied at runtime via the mapping above.