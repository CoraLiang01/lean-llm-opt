**Mathematical Model**

Let  

* \(W=\{S1,S2,S3,S4,S5\}\) – set of warehouses (rows of *supply_capacity.csv*).  
* \(C=\{D1,D2,D3,D4,D5\}\) – set of stores (rows of *customer_demand.csv*).  

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{daily demand of store }c &&\forall c\in C \\
s_w &\;=\;\text{daily supply capacity of warehouse }w &&\forall w\in W \\
c_{w,c} &\;=\;\text{transportation cost per unit from }w\text{ to }c &&\forall w\in W,\;c\in C
\end{aligned}
\]

Decision variables  

\[
x_{w,c}\;\ge 0 \quad\text{(units shipped from }w\text{ to }c)
\]

Objective  

\[
\min \; Z = \sum_{w\in W}\sum_{c\in C} c_{w,c}\,x_{w,c}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{w\in W} x_{w,c} = d_c \qquad \forall c\in C
\]

2. **Warehouse capacity**  
\[
\sum_{c\in C} x_{w,c} \le s_w \qquad \forall w\in W
\]

3. **Non‑negativity**  
\[
x_{w,c}\ge 0 \qquad \forall w\in W,\;c\in C
\]

---

### Data Mapping

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(d_c\) | `file_0_view_0` | `demand` | For each record, map `customer` → \(c\). |
| \(s_w\) | `file_1_view_0` | `supply_capacity` | For each record, map `region` → \(w\). |
| \(c_{w,c}\) | `file_2_view_0` | `D1`–`D5` | For each row, `Unnamed: 0` → \(w\); each column header `Dk` → \(c\). |

All index sets \(W\) and \(C\) are taken from the full set of rows in the corresponding tables; no additional filtering is applied. The model uses continuous shipment quantities, consistent with the problem statement.