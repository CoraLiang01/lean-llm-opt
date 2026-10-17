**Mathematical Model**

Let  

- \(S=\{S1,S2,S3,S4,S5\}\) be the set of warehouses (suppliers).  
- \(C=\{D1,D2,D3,D4,D5\}\) be the set of stores (customers).  

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{demand of store }c, &&c\in C,\\
s_s &\;=\;\text{supply capacity of warehouse }s, &&s\in S,\\
c_{s,c} &\;=\;\text{transportation cost per unit from }s\text{ to }c, &&s\in S,\;c\in C.
\end{aligned}
\]

Decision variables  

\[
x_{s,c}\;\in\;\mathbb{Z}_{\ge 0}\qquad (s\in S,\;c\in C)
\]

*Objective*  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

*Constraints*  

1. **Demand satisfaction**  
   \[
   \sum_{s\in S} x_{s,c}=d_c \qquad \forall\,c\in C
   \]

2. **Supply capacity**  
   \[
   \sum_{c\in C} x_{s,c}\le s_s \qquad \forall\,s\in S
   \]

3. **Non‑negativity & integrality**  
   \[
   x_{s,c}\ge 0,\; x_{s,c}\in\mathbb{Z}\qquad \forall\,s\in S,\;c\in C
   \]

---

**Data Mapping**

| Parameter | Source Table | Key Column(s) | Value Column |
|-----------|--------------|---------------|--------------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` |
| \(s_s\) | `file_1_view_0` | `region` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (row) = `s` ; column = `c` | cell value |

*Notes*  
- The transportation cost matrix `file_2_view_0` is 5 × 5, with rows identified by `Unnamed: 0` (warehouse) and columns by the store identifiers `D1`–`D5`.  
- All parameters are taken directly from the supplied CSV records; no additional data are introduced.