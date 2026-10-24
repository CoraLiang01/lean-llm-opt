**Mathematical model**

Let  

* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\) – workshops  
* \(C=\{C1,C2,\dots ,C111\}\) – component types  

Decision variable  

\[
y_c \;\ge 0 \qquad \forall c\in C
\]

*\(y_c\)* = number of units of component \(c\) produced in the plant.

Parameters  

\[
t_{w,c}\; \ge 0 \qquad \forall w\in W,\;c\in C
\]
*time (hours) required for one unit of component \(c\) in workshop \(w\)*  

\[
p_c \; \ge 0 \qquad \forall c\in C
\]
*unit selling price of component \(c\)*  

\[
H_w \; \ge 0 \qquad \forall w\in W
\]
*total available working hours in workshop \(w\)*  

---

### Objective  

\[
\max \;\; \sum_{c\in C} p_c\,y_c
\]

### Constraints  

\[
\sum_{c\in C} t_{w,c}\,y_c \;\le\; H_w \qquad \forall w\in W
\]

\[
y_c \;\ge\; 0 \qquad \forall c\in C
\]

---

**Data‑mapping rules**

| Parameter | Source table | Mapping rule |
|-----------|--------------|--------------|
| \(t_{w,c}\) | `file_0_view_0` (processing_time_unit.csv) | Row identifier `Unnamed: 0` → workshop \(w\); column identifier `Ck` → component \(c\). |
| \(p_c\) | `file_1_view_0` (unit_price.csv) | Row identifier `Unnamed: 0` → component \(c\); column `unit_price` → value. |
| \(H_w\) | `file_2_view_0` (total_working_hours.csv) | Row identifier `workshop` → workshop \(w\); column `total_hours` → value. |

All indices \(w\) and \(c\) are taken from the complete set of identifiers returned in the observation (5 workshops, 111 components). No additional constraints or bounds are imposed beyond those derived from the data.