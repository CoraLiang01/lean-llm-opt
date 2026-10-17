## Mathematical Model

**Index sets**  
\(S=\{s_1,\dots ,s_8\}\) – suppliers (from *supply_capacity.csv*)  
\(C=\{c_1,\dots ,c_8\}\) – customers (from *customer_demand.csv*)

**Parameters**  
\(d_c\) – daily demand of customer \(c\) (from *customer_demand.csv*)  
\(u_s\) – daily supply capacity of supplier \(s\) (from *supply_capacity.csv*)  
\(c_{s,c}\) – transportation cost per unit from supplier \(s\) to customer \(c\) (from *transportation_costs.csv*)

**Decision variables**  
\(x_{s,c}\ge 0\) – quantity shipped from supplier \(s\) to customer \(c\) (continuous)

**Objective**  
\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

**Constraints**

Demand satisfaction  
\[
\sum_{s\in S} x_{s,c}=d_c \qquad \forall\,c\in C
\]

Supply capacity  
\[
\sum_{c\in C} x_{s,c}\le u_s \qquad \forall\,s\in S
\]

Non‑negativity  
\[
x_{s,c}\ge 0 \qquad \forall\,s\in S,\;c\in C
\]

---

## Data Mapping

| Parameter | Source Table ID | Column(s) | Mapping Notes |
|-----------|-----------------|-----------|---------------|
| \(d_c\) | `file_0_view_0` | `demand` | Row axis: `Customers` |
| \(u_s\) | `file_1_view_0` | `supply_capacity` | Row axis: `Supplier` |
| \(c_{s,c}\) | `file_2_view_0` | `demand1` … `demand8` | Matrix: rows `Unnamed: 0` (suppliers), columns `Customers` |

*All indices \(s\) and \(c\) are taken from the full set of rows/columns returned in the observation; no additional filtering is applied.*