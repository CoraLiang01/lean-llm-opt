**Mathematical Model**

Let  

* \(W=\{W1,\dots ,W10\}\) – set of warehouses (index from *file_1_view_0* “Warehouse ID”).  
* \(C=\{C1,\dots ,C20\}\) – set of customers (index from *file_2_view_0* “Customer ID”).  

Parameters  

* \(f_w\) – fixed annual opening cost of warehouse \(w\)  
  \(\displaystyle f_w =\) *file_1_view_0* column **Fixed_Cost**.  
* \(s_w\) – capacity of warehouse \(w\)  
  \(\displaystyle s_w =\) *file_1_view_0* column **Capacity**.  
* \(d_c\) – demand of customer \(c\)  
  \(\displaystyle d_c =\) *file_2_view_0* column **Demand**.  
* \(c_{wc}\) – unit transportation cost from warehouse \(w\) to customer \(c\)  
  \(\displaystyle c_{wc} =\) *file_0_view_0* matrix entry at row \(w\), column \(c\).  

Decision variables  

* \(y_w \in \{0,1\}\) – 1 if warehouse \(w\) is opened, 0 otherwise.  
* \(x_{wc}\ge 0\) – quantity shipped from \(w\) to \(c\).  

\[
\begin{aligned}
\min\;& \sum_{w\in W} f_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} c_{wc}\,x_{wc} \\[4pt]
\text{s.t. } & \sum_{w\in W} x_{wc} \;\ge\; d_c , && \forall c\in C \quad\text{(demand)}\\[4pt]
& \sum_{c\in C} x_{wc} \;\le\; s_w\,y_w , && \forall w\in W \quad\text{(capacity)}\\[4pt]
& x_{wc}\;\ge\;0 , && \forall w\in W,\;c\in C\\
& y_w\in\{0,1\} , && \forall w\in W
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column / Matrix Cell | Notes |
|-----------|--------------|----------------------|-------|
| \(f_w\) | *file_1_view_0* | **Fixed_Cost** | Warehouse ID matches index \(w\). |
| \(s_w\) | *file_1_view_0* | **Capacity** | Warehouse ID matches index \(w\). |
| \(d_c\) | *file_2_view_0* | **Demand** | Customer ID matches index \(c\). |
| \(c_{wc}\) | *file_0_view_0* | Matrix entry at row \(w\), column \(c\) | Row ID = Warehouse ID, column ID = Customer ID. |

Index sets are derived directly from the unique identifiers in the respective tables:

* \(W =\) set of all **Warehouse ID** values in *file_1_view_0* (10 elements).  
* \(C =\) set of all **Customer ID** values in *file_2_view_0* (20 elements).  

No additional data or assumptions are introduced beyond what is supplied in the tables.