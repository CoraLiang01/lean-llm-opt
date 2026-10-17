## Mathematical Model

**Index sets**  
\(W=\{S1,\dots ,S10\}\) (warehouses)  
\(C=\{C1,\dots ,C10\}\) (customers)

**Parameters**  
\(d_c\) daily demand of customer \(c\) (from *customer_demand.csv*)  
\(s_w\) daily supply capacity of warehouse \(w\) (from *supply_capacity.csv*)  
\(c_{w,c}\) transportation cost per unit from warehouse \(w\) to customer \(c\) (from *transportation_costs.csv*)

**Decision variables**  
\(x_{w,c}\ge 0\) units shipped from \(w\) to \(c\)

**Objective**  
\[
\min \;\sum_{w\in W}\sum_{c\in C} c_{w,c}\,x_{w,c}
\]

**Constraints**  

Demand satisfaction  
\[
\sum_{w\in W} x_{w,c}\;\ge\; d_c \qquad \forall\,c\in C
\]

Supply limits  
\[
\sum_{c\in C} x_{w,c}\;\le\; s_w \qquad \forall\,w\in W
\]

Non‑negativity  
\[
x_{w,c}\;\ge\;0 \qquad \forall\,w\in W,\;c\in C
\]

---

## Data Mapping

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(d_c\) | `file_0_view_0` | `demand` | row identified by `customer` |
| \(s_w\) | `file_1_view_0` | `supply_capacity` | row identified by `Unnamed: 0` |
| \(c_{w,c}\) | `file_2_view_0` | column `C1`–`C10` | row identified by `Unnamed: 0` |

All index sets \(W\) and \(C\) are taken from the distinct values present in the corresponding source tables. No additional entities are introduced beyond those listed.