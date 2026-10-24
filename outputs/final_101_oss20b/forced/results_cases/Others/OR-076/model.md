**Mathematical Model**

Let  

* \(W=\{W1,\dots ,W10\}\) – set of warehouses  
* \(C=\{C1,\dots ,C20\}\) – set of customers  

Parameters  

* \(f_w\) – fixed annual opening cost of warehouse \(w\)  (from *warehouse.csv*, column **Fixed_Cost**)  
* \(cap_w\) – maximum service capacity of warehouse \(w\)  (from *warehouse.csv*, column **Capacity**)  
* \(d_c\) – demand of customer \(c\)  (from *demand.csv*, column **Demand**)  
* \(c_{wc}\) – unit transportation cost from warehouse \(w\) to customer \(c\)  (from *cost.csv*, matrix entry \((w,c)\))

Decision variables  

* \(y_w \in \{0,1\}\) – 1 if warehouse \(w\) is opened, 0 otherwise  
* \(x_{wc}\ge 0\) – quantity shipped from \(w\) to \(c\)

Objective  

\[
\min \;\sum_{w\in W} f_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} c_{wc}\,x_{wc}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{w\in W} x_{wc}=d_c \qquad \forall\,c\in C
\]

2. **Warehouse capacity**  
\[
\sum_{c\in C} x_{wc}\;\le\; cap_w\,y_w \qquad \forall\,w\in W
\]

3. **Non‑negativity**  
\[
x_{wc}\ge 0 \qquad \forall\,w\in W,\;c\in C
\]

4. **Binary opening decision**  
\[
y_w\in\{0,1\}\qquad \forall\,w\in W
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(f_w\) | *warehouse.csv* | **Fixed_Cost** | \(f_{W_i}= \text{value of Fixed_Cost for Warehouse ID }W_i\) |
| \(cap_w\) | *warehouse.csv* | **Capacity** | \(cap_{W_i}= \text{value of Capacity for Warehouse ID }W_i\) |
| \(d_c\) | *demand.csv* | **Demand** | \(d_{C_j}= \text{value of Demand for Customer ID }C_j\) |
| \(c_{wc}\) | *cost.csv* | matrix entry \((W_i,C_j)\) | \(c_{W_iC_j}= \text{value in row }W_i,\text{ column }C_j\) |

*Index sets*  
- \(W\) is the set of all **Warehouse ID** values in *warehouse.csv* (10 elements).  
- \(C\) is the set of all **Customer ID** values in *demand.csv* (20 elements).  

All parameters are taken directly from the corresponding source tables; no additional processing or enumeration of values is required in the model.