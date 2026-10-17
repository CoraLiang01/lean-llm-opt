## Mathematical Model

**Index sets**  
\(W\) – set of warehouses (Warehouse ID from *file_1_view_0*).  
\(C\) – set of customers (Customer ID from *file_2_view_0*).

**Parameters**  
\(f_w\) – fixed annual opening cost of warehouse \(w\) (column *Fixed_Cost* in *file_1_view_0*).  
\(cap_w\) – maximum service capacity of warehouse \(w\) (column *Capacity* in *file_1_view_0*).  
\(d_c\) – demand of customer \(c\) (column *Demand* in *file_2_view_0*).  
\(t_{w,c}\) – unit transportation cost from warehouse \(w\) to customer \(c\) (matrix *file_0_view_0*; row axis *Warehouse ID*, column axis *Customer ID*).

**Decision variables**  
\(y_w \in \{0,1\}\) – 1 if warehouse \(w\) is opened, 0 otherwise.  
\(x_{w,c} \ge 0\) – quantity shipped from \(w\) to \(c\).

**Objective**  
\[
\min \;\sum_{w\in W} f_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} t_{w,c}\,x_{w,c}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{w\in W} x_{w,c} = d_c \quad \forall\,c\in C
\]

2. **Warehouse capacity**  
\[
\sum_{c\in C} x_{w,c} \le cap_w\,y_w \quad \forall\,w\in W
\]

3. **Non‑negativity**  
\[
x_{w,c} \ge 0 \quad \forall\,w\in W,\;c\in C
\]

4. **Binary opening decision**  
\[
y_w \in \{0,1\} \quad \forall\,w\in W
\]

---

## Data Mapping Rules

| Parameter | Source Table | Key Columns | Value Column | Mapping |
|-----------|--------------|-------------|--------------|---------|
| \(f_w\) | *file_1_view_0* | Warehouse ID | Fixed_Cost | \(f_{Warehouse\,ID}\) |
| \(cap_w\) | *file_1_view_0* | Warehouse ID | Capacity | \(cap_{Warehouse\,ID}\) |
| \(d_c\) | *file_2_view_0* | Customer ID | Demand | \(d_{Customer\,ID}\) |
| \(t_{w,c}\) | *file_0_view_0* | Warehouse ID (row) | Customer ID (column) | \(t_{Warehouse\,ID,Customer\,ID}\) |

- **Index sets** \(W\) and \(C\) are the full sets of Warehouse ID and Customer ID values present in the respective tables.
- All numeric values are taken directly from the specified columns; no transformations or aggregations are applied.
- The matrix *file_0_view_0* is guaranteed to be a complete \(10 \times 20\) matrix, so every pair \((w,c)\) has a defined transportation cost.

This model fully captures the problem statement and uses only the data provided in the current observation.