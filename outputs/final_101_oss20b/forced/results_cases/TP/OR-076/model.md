**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{w\in W}\bigl(\text{fixed\_cost}_w\,y_w
      +\sum_{c\in C}\text{cost}_{w,c}\,x_{w,c}\bigr)\\[4pt]
\text{s.t. }&\sum_{w\in W}x_{w,c}\;\ge\;\text{demand}_c,
      &&\forall\,c\in C\\[4pt]
&\sum_{c\in C}x_{w,c}\;\le\;\text{capacity}_w\,y_w,
      &&\forall\,w\in W\\[4pt]
&x_{w,c}\;\ge\;0,\qquad y_w\in\{0,1\},
      &&\forall\,w\in W,\;c\in C
\end{aligned}
\]

*Sets*  
\(W=\{W1,\dots ,W10\}\) – warehouses (from **Warehouse ID** column of *file_1_view_0*)  
\(C=\{C1,\dots ,C20\}\) – customers (from **Customer ID** column of *file_2_view_0*)

*Parameters*  

| Parameter | Source Table | Column (or matrix cell) |
|-----------|--------------|------------------------|
| \(\text{fixed\_cost}_w\) | *file_1_view_0* | **Fixed_Cost** |
| \(\text{capacity}_w\) | *file_1_view_0* | **Capacity** |
| \(\text{demand}_c\) | *file_2_view_0* | **Demand** |
| \(\text{cost}_{w,c}\) | *file_0_view_0* | matrix cell at row **Warehouse ID** = \(w\), column **Customer ID** = \(c\) |

*Decision Variables*  

- \(y_w\in\{0,1\}\): 1 if warehouse \(w\) is opened, 0 otherwise.  
- \(x_{w,c}\ge0\): quantity shipped from warehouse \(w\) to customer \(c\).

---

**Data Mapping Rules**

1. **Warehouse ID** → index set \(W\).  
2. **Customer ID** → index set \(C\).  
3. For each warehouse \(w\):  
   - `fixed_cost_w` ← *file_1_view_0* column **Fixed_Cost**.  
   - `capacity_w` ← *file_1_view_0* column **Capacity**.  
4. For each customer \(c\):  
   - `demand_c` ← *file_2_view_0* column **Demand**.  
5. For each pair \((w,c)\):  
   - `cost_{w,c}` ← *file_0_view_0* matrix cell where row label equals `Warehouse ID` \(w\) and column label equals `Customer ID` \(c\).

These mappings ensure every parameter used in the model is directly traceable to the source data.