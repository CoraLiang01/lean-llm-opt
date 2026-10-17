**Mathematical Model**

Let  

* \(S=\{S1,\dots ,S12\}\) – supply‑center identifiers (row id “Unnamed: 0” in *supply_capacity.csv*).  
* \(C=\{C1,\dots ,C12\}\) – customer‑group identifiers (column id “customer” in *customer_demand.csv*).  

Parameters  

* \(d_c\) – demand of customer \(c\) (column “demand” in *customer_demand.csv*).  
* \(s_s\) – supply capacity of center \(s\) (column “supply_capacity” in *supply_capacity.csv*).  
* \(c_{s,c}\) – transportation cost per unit from center \(s\) to customer \(c\) (matrix entry in *transportation_costs.csv*).  

Decision variables  

\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

---

### Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

### Constraints  

**Supply‑capacity constraints**  
\[
\sum_{c\in C} x_{s,c}\;\le\; s_s \qquad \forall s\in S
\]

**Demand‑satisfaction constraints**  
\[
\sum_{s\in S} x_{s,c}\;\ge\; d_c \qquad \forall c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column(s) | Value Column |
|-----------|--------------|---------------|--------------|
| \(d_c\) | `file_0_view_0` | `customer` | `demand` |
| \(s_s\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | Row key: `Unnamed: 0` (supply center) <br> Column key: `C1`–`C12` (customer) | cost value (numeric) |

All indices \(S\) and \(C\) are taken directly from the corresponding key columns in the source tables. No additional data or assumptions are introduced beyond what is supplied.