## Mathematical Model

**Index sets**  
\(S=\{S1,\dots ,S10\}\) (suppliers)  
\(C=\{C1,\dots ,C10\}\) (customers)

**Parameters**  
\(d_c\) : demand of customer \(c\) (from *customer_demand.csv*)  
\(s_s\) : supply capacity of supplier \(s\) (from *supply_capacity.csv*)  
\(c_{s,c}\) : transportation cost per unit from supplier \(s\) to customer \(c\) (from *transportation_costs.csv*)

**Decision variables**  
\(x_{s,c}\ge 0\) : quantity shipped from supplier \(s\) to customer \(c\) (continuous)

**Objective**  
\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

**Constraints**

Supply limits  
\[
\sum_{c\in C} x_{s,c}\;\le\; s_s \qquad \forall s\in S
\]

Demand satisfaction  
\[
\sum_{s\in S} x_{s,c}\;=\; d_c \qquad \forall c\in C
\]

Non‑negativity  
\[
x_{s,c}\;\ge\;0 \qquad \forall s\in S,\;c\in C
\]

---

## Data Mapping

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(d_c\) | `file_0_view_0` | `demand` | `customer` → index \(c\) |
| \(s_s\) | `file_1_view_0` | `supply_capacity` | `Unnamed: 0` → index \(s\) |
| \(c_{s,c}\) | `file_2_view_0` | `C1`–`C10` | `Unnamed: 0` → index \(s\), column name → index \(c\) |

All indices \(s\) and \(c\) are taken from the full set of rows in the respective tables; no additional filtering is applied. The model uses the exact numeric values from these tables.