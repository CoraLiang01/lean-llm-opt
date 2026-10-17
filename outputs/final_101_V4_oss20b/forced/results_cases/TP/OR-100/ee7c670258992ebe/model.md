**Mathematical model**

Let  

* \(C=\{C1,\dots ,C111\}\) (components)  
* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly}\}\) (workshops)

Decision variables  
\[
x_c\;\ge 0 \qquad \forall c\in C
\]

Parameters  

\[
\text{price}_c \quad\text{(unit price of component }c\text{)}  
\]
\[
\text{time}_{w,c}\quad\text{(hours required in workshop }w\text{ for one unit of }c)  
\]
\[
\text{total\_hours}_w\quad\text{(available hours in workshop }w)
\]

Objective  

\[
\max \;\sum_{c\in C}\text{price}_c\,x_c
\]

Constraints  

\[
\sum_{c\in C}\text{time}_{w,c}\,x_c \;\le\; \text{total\_hours}_w
\qquad \forall w\in W
\]

---

**Data‑mapping rules**

| Parameter | Source table | Source columns | Mapping rule |
|-----------|--------------|----------------|--------------|
| `price_c` | `file_1_view_0` (`unit_price.csv`) | `Unnamed: 0` → component id, `unit_price` → value | `price[c] = unit_price` where `c = Unnamed: 0` |
| `time_{w,c}` | `file_0_view_0` (`processing_time_unit.csv`) | `Unnamed: 0` → workshop id, `C1`–`C111` → times | `time[w,c] = value` where `w = Unnamed: 0` and `c = column name` |
| `total_hours_w` | `file_2_view_0` (`total_working_hours.csv`) | `workshop` → workshop id, `total_hours` → value | `total_hours[w] = total_hours` where `w = workshop` |

All indices and parameters are taken directly from the retrieved data; no additional data are introduced.