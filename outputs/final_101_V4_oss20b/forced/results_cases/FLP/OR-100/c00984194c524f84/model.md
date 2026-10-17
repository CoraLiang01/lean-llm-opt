**Mathematical Model**

Let  

* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\)  
* \(C=\{C1,C2,\dots ,C111\}\)

Decision variables  
\[
x_c \;\ge 0 \qquad \forall c\in C
\]

Parameters  
\[
\text{time}_{w,c}\quad\text{(hours per unit of component }c\text{ in workshop }w)\\
\text{price}_c\quad\text{(unit selling price of component }c)\\
\text{total\_hours}_w\quad\text{(available working hours in workshop }w)
\]

Objective (maximise total output value)  
\[
\max \;\sum_{c\in C}\text{price}_c\,x_c
\]

Workshop‑capacity constraints  
\[
\sum_{c\in C}\text{time}_{w,c}\,x_c \;\le\; \text{total\_hours}_w
\qquad \forall w\in W
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Row Axis | Column Axis | Column Name |
|-----------|--------------|----------|-------------|-------------|
| \(\text{time}_{w,c}\) | `file_0_view_0` | `Unnamed: 0` (workshop) | `C1`–`C111` (component) | value |
| \(\text{price}_c\) | `file_1_view_0` | `Unnamed: 0` (component) | `unit_price` | value |
| \(\text{total\_hours}_w\) | `file_2_view_0` | `workshop` | `total_hours` | value |

*Note:* The entry for workshop **Finishing** is missing in `file_2_view_0`; the model requires this value to be supplied for the corresponding capacity constraint.