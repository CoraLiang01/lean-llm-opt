**Mathematical Model**

Let  

- \(W=\{S1,S2,S3,S4,S5\}\) (warehouses)  
- \(C=\{D1,D2,D3,D4,D5\}\) (stores)

Parameters  

- \(d_c\) : daily demand of store \(c\) (from **file_0_view_0**, column *demand*)  
- \(s_w\) : daily supply capacity of warehouse \(w\) (from **file_1_view_0**, column *supply_capacity*)  
- \(c_{w,c}\) : transportation cost per unit from warehouse \(w\) to store \(c\) (from **file_2_view_0**, row *Unnamed: 0* = \(w\), column \(c\))

Decision variables  

- \(x_{w,c}\ge 0\) units shipped from \(w\) to \(c\) (\(w\in W,\;c\in C\))

Objective  

\[
\min \;\sum_{w\in W}\sum_{c\in C} c_{w,c}\,x_{w,c}
\]

Constraints  

1. **Demand satisfaction** for every store \(c\in C\)

\[
\sum_{w\in W} x_{w,c}= d_c
\]

2. **Supply capacity** for every warehouse \(w\in W\)

\[
\sum_{c\in C} x_{w,c}\le s_w
\]

3. **Non‑negativity**

\[
x_{w,c}\ge 0 \qquad \forall w\in W,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(d_c\) | **file_0_view_0** | customer | demand |
| \(s_w\) | **file_1_view_0** | region | supply_capacity |
| \(c_{w,c}\) | **file_2_view_0** | Unnamed: 0 (row id) | D1–D5 (columns) |

All index sets \(W\) and \(C\) are taken from the row and column identifiers of the respective tables. No additional filtering or aggregation is applied.