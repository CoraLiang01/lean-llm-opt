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

Subject to  

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

| Parameter | Source Table | Source Columns | Mapping Description |
|-----------|--------------|----------------|----------------------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | Each row gives the demand for a customer. |
| \(s_s\) | `file_1_view_0` | `Unnamed: 0` (supplier id), `supply_capacity` | Each row gives the supply capacity of a supplier. |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (supplier id) as row key, columns `C1`–`C12` as customer keys | Each cell gives the unit transportation cost from the supplier in the row to the customer in the column. |

All sets \(S\) and \(C\) are derived from the distinct values of the supplier and customer identifiers in the respective tables. No numerical values are included in the model; they are supplied via the data mapping above.