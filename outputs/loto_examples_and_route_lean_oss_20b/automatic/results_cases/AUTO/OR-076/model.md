**Mathematical Model**

Let  

- \(W=\{W1,\dots ,W10\}\) – set of warehouses (Warehouse ID from *file_1_view_0*).  
- \(C=\{C1,\dots ,C20\}\) – set of customers (Customer ID from *file_2_view_0*).  

Parameters  

- \(f_w\) – fixed annual opening cost of warehouse \(w\) (column *Fixed_Cost* in *file_1_view_0*).  
- \(cap_w\) – maximum service capacity of warehouse \(w\) (column *Capacity* in *file_1_view_0*).  
- \(d_c\) – demand of customer \(c\) (column *Demand* in *file_2_view_0*).  
- \(t_{w,c}\) – unit transportation cost from warehouse \(w\) to customer \(c\) (matrix *file_0_view_0*; row axis *Warehouse ID*, column axis *Customer ID*).  

Decision variables  

- \(y_w\in\{0,1\}\) – 1 if warehouse \(w\) is opened, 0 otherwise.  
- \(x_{w,c}\ge 0\) – quantity shipped from \(w\) to \(c\).  

\[
\begin{aligned}
\min\;&\sum_{w\in W} f_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} t_{w,c}\,x_{w,c} \\[4pt]
\text{s.t. } &\sum_{w\in W} x_{w,c}= d_c , &&\forall c\in C \quad\text{(demand satisfaction)}\\[4pt]
&\sum_{c\in C} x_{w,c}\;\le\; cap_w\,y_w , &&\forall w\in W \quad\text{(capacity)}\\[4pt]
&x_{w,c}\;\ge\;0 , &&\forall w\in W,\;c\in C\\
&y_w\in\{0,1\} , &&\forall w\in W
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column(s) | Mapping |
|-----------|--------|----------|-----------|---------|
| \(f_w\) | Fixed cost | *file_1_view_0* | *Fixed_Cost* | one‑to‑one with *Warehouse ID* |
| \(cap_w\) | Capacity | *file_1_view_0* | *Capacity* | one‑to‑one with *Warehouse ID* |
| \(d_c\) | Demand | *file_2_view_0* | *Demand* | one‑to‑one with *Customer ID* |
| \(t_{w,c}\) | Transportation cost | *file_0_view_0* | matrix entries | row axis *Warehouse ID*, column axis *Customer ID* |

All indices \(w\) and \(c\) are taken from the respective ID columns of the source tables. No additional filtering or aggregation is applied. The model uses the full sets of warehouses and customers as provided.