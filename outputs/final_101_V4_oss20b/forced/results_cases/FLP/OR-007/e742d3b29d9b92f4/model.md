**Mathematical Model**

Let  

- \(I=\{S1,S2,S3,S4,S5\}\) (warehouses)  
- \(J=\{D1,D2,D3,D4,D5\}\) (stores)  

Decision variables  
\[
x_{ij}\;\ge 0 \qquad \forall i\in I,\; j\in J
\]

Parameters  
\[
d_j \quad\text{(daily demand of store }j) \\
s_i \quad\text{(daily supply capacity of warehouse }i) \\
c_{ij}\quad\text{(unit transportation cost from }i\text{ to }j)
\]

Objective  
\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J
\]

2. **Supply capacity**  
\[
\sum_{j\in J} x_{ij}\le s_i \qquad \forall i\in I
\]

3. **Non‑negativity**  
\[
x_{ij}\ge 0 \qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping Rules**

| Parameter | Source Table ID | Row Identifier | Column Identifier |
|-----------|-----------------|----------------|-------------------|
| \(d_j\)   | `file_0_view_0` | `customer`     | `demand`          |
| \(s_i\)   | `file_1_view_0` | `region`       | `supply_capacity` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0`   | `D1`–`D5` (columns) |

*The cost matrix \(c_{ij}\) is a 5 × 5 matrix with rows identified by the `Unnamed: 0` column and columns `D1` through `D5`.*