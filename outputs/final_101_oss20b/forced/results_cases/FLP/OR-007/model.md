**Mathematical Model**

Let  

- \(I=\{S1,S2,S3,S4,S5\}\) (warehouses)  
- \(J=\{D1,D2,D3,D4,D5\}\) (stores)

Parameters  

- \(d_j\) : daily demand of store \(j\)  \(j\in J\)  
- \(s_i\) : daily supply capacity of warehouse \(i\) \(i\in I\)  
- \(c_{ij}\) : unit transportation cost from warehouse \(i\) to store \(j\) \(i\in I,\;j\in J\)

Decision variables  

- \(x_{ij}\ge 0\) : quantity shipped from warehouse \(i\) to store \(j\)

Objective  

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j,\qquad \forall j\in J
   \]

2. **Supply capacity**  
   \[
   \sum_{j\in J} x_{ij}\le s_i,\qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\ge 0,\qquad \forall i\in I,\;j\in J
   \]

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column | Mapping Expression |
|-----------|--------------|------------|--------------|---------------------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | `d_j = demand where customer = j` |
| \(s_i\) | `file_1_view_0` | `region` | `supply_capacity` | `s_i = supply_capacity where region = i` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `D1`–`D5` | `c_{ij} = value in column D_j where Unnamed: 0 = i` |

All indices \(i\) and \(j\) are taken from the full set of rows/columns present in the respective source tables. No additional constraints or variables are introduced beyond those specified.