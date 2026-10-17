**Mathematical Model**

Let  

* \(S=\{S_1,S_2,S_3,S_4,S_5\}\) – warehouses (regions)  
* \(D=\{D_1,D_2,D_3,D_4,D_5\}\) – stores (customers)  

Parameters  

* \(c_{s,d}\) – transportation cost per unit from warehouse \(s\) to store \(d\)  
* \(d_d\) – daily demand of store \(d\)  
* \(cap_s\) – daily supply capacity of warehouse \(s\)

Decision variables  

* \(x_{s,d}\ge 0\) – quantity shipped from \(s\) to \(d\)

\[
\begin{aligned}
\min\;& \sum_{s\in S}\sum_{d\in D} c_{s,d}\,x_{s,d} \\[4pt]
\text{s.t.}\;& \sum_{s\in S} x_{s,d}= d_d \qquad \forall d\in D \\[4pt]
& \sum_{d\in D} x_{s,d}\le cap_s \qquad \forall s\in S \\[4pt]
& x_{s,d}\ge 0 \qquad \forall s\in S,\; d\in D
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Row Identifier | Column Identifier | Notes |
|-----------|--------------|----------------|-------------------|-------|
| \(d_d\) | `file_0_view_0` | `customer` | `demand` | Demand for each store |
| \(cap_s\) | `file_1_view_0` | `region` | `supply_capacity` | Capacity of each warehouse |
| \(c_{s,d}\) | `file_2_view_0` | `Unnamed: 0` (warehouse) | `D1`–`D5` (stores) | Cost per unit from warehouse to store |

The mapping preserves the exact identifiers and column names from the CSV files; no numerical values are enumerated in the model.