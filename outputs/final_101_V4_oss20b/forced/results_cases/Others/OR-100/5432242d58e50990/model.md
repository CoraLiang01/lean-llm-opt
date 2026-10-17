## Mathematical Model

**Sets**  
\(C=\{C1,\dots ,C111\}\) components  
\(W=\{\text{Casting},\text{Milling},\text{Assembly},\text{QA \& Packaging}\}\) workshops  

**Parameters**  
\(t_{w,c}\) – processing time (hours) of one unit of component \(c\) in workshop \(w\)  
\(p_c\) – unit price (yuan) of component \(c\)  
\(H_w\) – total available working hours in workshop \(w\)

**Decision Variables**  
\(x_c\ge 0\) production quantity of component \(c\)

**Objective**  
\[
\max \;\sum_{c\in C} p_c\,x_c
\]

**Constraints**  
For every workshop \(w\in W\):
\[
\sum_{c\in C} t_{w,c}\,x_c \;\le\; H_w
\]

---

## Data Mapping

| Parameter | Source Table | Key Columns | Value Column |
|-----------|--------------|-------------|--------------|
| \(t_{w,c}\) | `file_0_view_0` (processing_time_unit.csv) | `Unnamed: 0` = workshop \(w\); column `Ck` = component \(c\) | cell value |
| \(p_c\) | `file_1_view_0` (unit_price.csv) | `Unnamed: 0` = component \(c\) | `unit_price` |
| \(H_w\) | `file_2_view_0` (total_working_hours.csv) | `workshop` = workshop \(w\) | `total_hours` |

*Note:* The Finishing workshop appears in the processing‑time table but has no corresponding entry in the total‑working‑hours table; therefore no capacity constraint is generated for it. All other workshops listed in the capacity table are included.