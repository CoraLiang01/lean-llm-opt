**Mathematical Model**

Let  

- \(W\) – set of warehouses (indices \(w\)),  
- \(C\) – set of customers (indices \(c\)).  

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_w\) | Fixed annual opening cost of warehouse \(w\) | `file_1_view_0`, column `Fixed_Cost` |
| \(cap_w\) | Maximum service capacity of warehouse \(w\) | `file_1_view_0`, column `Capacity` |
| \(d_c\) | Demand of customer \(c\) | `file_2_view_0`, column `Demand` |
| \(t_{w,c}\) | Variable transportation cost per unit from warehouse \(w\) to customer \(c\) | `file_0_view_0`, matrix entry (row `Warehouse ID` = \(w\), column `Customer ID` = \(c\)) |

**Decision Variables**

- \(y_w \in \{0,1\}\) – 1 if warehouse \(w\) is opened, 0 otherwise.  
- \(x_{w,c} \ge 0\) – quantity shipped from warehouse \(w\) to customer \(c\).

**Objective**

\[
\min \;\; \sum_{w\in W} f_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} t_{w,c}\,x_{w,c}
\]

**Constraints**

1. **Capacity limits**

\[
\sum_{c\in C} x_{w,c} \;\le\; cap_w \, y_w \quad \forall w\in W
\]

2. **Demand satisfaction**

\[
\sum_{w\in W} x_{w,c} \;=\; d_c \quad \forall c\in C
\]

3. **Variable domains**

\[
y_w \in \{0,1\}\quad \forall w\in W,\qquad
x_{w,c} \ge 0 \quad \forall w\in W,\;c\in C
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Mapping Rule |
|-----------|----------|-----------|--------------|
| \(f_w\) | `file_1_view_0` | `Fixed_Cost` | `f_w = value where Warehouse ID = w` |
| \(cap_w\) | `file_1_view_0` | `Capacity` | `cap_w = value where Warehouse ID = w` |
| \(d_c\) | `file_2_view_0` | `Demand` | `d_c = value where Customer ID = c` |
| \(t_{w,c}\) | `file_0_view_0` | matrix entry | `t_{w,c} = value at row Warehouse ID = w, column Customer ID = c` |

All index sets \(W\) and \(C\) are taken from the distinct `Warehouse ID` and `Customer ID` values present in the respective tables. No additional constraints or parameters are introduced beyond those specified.