**Mathematical Model**

Let  

- \(W=\{W1,\dots ,W10\}\) : set of potential warehouses  
- \(C=\{C1,\dots ,C20\}\) : set of customers  

Parameters  

- \(f_w\) : fixed annual opening cost of warehouse \(w\)  \(w\in W\)  
- \(cap_w\) : maximum service capacity of warehouse \(w\)  \(w\in W\)  
- \(d_c\) : demand of customer \(c\)  \(c\in C\)  
- \(t_{wc}\) : unit transportation cost from warehouse \(w\) to customer \(c\) \(w\in W,\;c\in C\)

Decision variables  

- \(y_w\in\{0,1\}\) : 1 if warehouse \(w\) is opened, 0 otherwise  
- \(x_{wc}\ge 0\) : quantity shipped from \(w\) to \(c\)

Objective  

\[
\min \;\sum_{w\in W} f_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} t_{wc}\,x_{wc}
\]

Constraints  

1. **Capacity (only if warehouse is open)**  
\[
\sum_{c\in C} x_{wc} \;\le\; cap_w\,y_w \qquad \forall w\in W
\]

2. **Demand satisfaction**  
\[
\sum_{w\in W} x_{wc} \;=\; d_c \qquad \forall c\in C
\]

3. **Variable domains**  
\[
x_{wc}\;\ge\;0 \quad \forall w\in W,\;c\in C
\]
\[
y_w\in\{0,1\}\quad \forall w\in W
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(f_w\)   | `file_1_view_0` | `Fixed_Cost` |
| \(cap_w\) | `file_1_view_0` | `Capacity` |
| \(d_c\)   | `file_2_view_0` | `Demand` |
| \(t_{wc}\)| `file_0_view_0` | matrix entry at row `Warehouse ID` = \(w\), column `Customer ID` = \(c\) |

Index sets are taken directly from the identifiers present in the source tables. No numerical values are enumerated in the model.