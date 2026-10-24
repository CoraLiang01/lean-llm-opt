**Mathematical Model**

Let  

* \(W\) – set of warehouses (indices \(w\)).  
* \(C\) – set of customers (indices \(c\)).  

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(F_w\) | Fixed annual opening cost of warehouse \(w\) | `file_1_view_0` column **Fixed_Cost** |
| \(K_w\) | Capacity of warehouse \(w\) | `file_1_view_0` column **Capacity** |
| \(D_c\) | Demand of customer \(c\) | `file_2_view_0` column **Demand** |
| \(t_{wc}\) | Transportation cost per unit from warehouse \(w\) to customer \(c\) | `file_0_view_0` column **C1**–**C20** (column **Cj** corresponds to customer \(c_j\)) |

**Decision Variables**

* \(y_w \in \{0,1\}\) – 1 if warehouse \(w\) is opened, 0 otherwise.  
* \(x_{wc} \ge 0\) – quantity shipped from warehouse \(w\) to customer \(c\).

**Objective**

\[
\min \;\; \sum_{w\in W} F_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} t_{wc}\,x_{wc}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{w\in W} x_{wc} \;=\; D_c , \qquad \forall c\in C
   \]

2. **Warehouse capacity (only if opened)**  
   \[
   \sum_{c\in C} x_{wc} \;\le\; K_w\,y_w , \qquad \forall w\in W
   \]

3. **Non‑negativity**  
   \[
   x_{wc} \;\ge\; 0 , \qquad \forall w\in W,\;c\in C
   \]

4. **Binary opening decision**  
   \[
   y_w \;\in\; \{0,1\} , \qquad \forall w\in W
   \]

---

### Data Mapping

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(F_w\) | `file_1_view_0` | `Fixed_Cost` |
| \(K_w\) | `file_1_view_0` | `Capacity` |
| \(D_c\) | `file_2_view_0` | `Demand` |
| \(t_{wc}\) | `file_0_view_0` | `Cj` where \(j\) is the position of customer \(c\) in the customer list (C1 for C1, C2 for C2, …, C20 for C20) |

*Index sets*  
- \(W\) is the set of all `Warehouse ID` values in `file_1_view_0`.  
- \(C\) is the set of all `Customer ID` values in `file_2_view_0`.  

This symbolic model fully captures the optimization problem described in the query.