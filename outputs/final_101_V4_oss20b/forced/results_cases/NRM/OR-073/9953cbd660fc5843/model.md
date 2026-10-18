**Abstract Model**

##### Index Sets
- \(P=\{I,II,III\}\) (products)  
- \(R=\{A,B\}\) (procedures)  
- \(E=\{A1,A2,B1,B2,B3\}\) (equipment types)

##### Parameters
- \(t_{p,r,e}\) processing time (hours) of product \(p\) on procedure \(r\) using equipment \(e\)  
- \(c^{RM}_p\) raw‑material cost per unit of product \(p\)  
- \(p_p\) selling price per unit of product \(p\)  
- \(T_e\) available operating time (hours) of equipment \(e\) over the horizon  
- \(c^E_e\) equipment cost per hour of equipment \(e\) at full load  

##### Decision Variables
- \(y_p\ge 0\) total production quantity of product \(p\) (continuous)  
- \(x_{p,r,e}\ge 0\) quantity of product \(p\) processed on procedure \(r\) using equipment \(e\) (continuous)

##### Objective Function
\[
\max \;\; \sum_{p\in P}\bigl(p_p\,y_p-c^{RM}_p\,y_p\bigr)
\;-\;\sum_{e\in E}\sum_{p\in P}\sum_{r\in R}c^E_e\,t_{p,r,e}\,x_{p,r,e}
\]

##### Constraints

###### 1. Flow balance for procedure A  
\[
\sum_{e\in E}x_{p,A,e}=y_p ,\qquad \forall p\in P
\]

###### 2. Flow balance for procedure B  
\[
\sum_{e\in E}x_{p,B,e}=y_p ,\qquad \forall p\in P
\]

###### 3. Equipment capacity constraints  
\[
\sum_{p\in P}\sum_{r\in R}t_{p,r,e}\,x_{p,r,e}\le T_e ,\qquad \forall e\in E
\]

###### 4. Non‑negativity  
\[
y_p\ge 0,\;\;x_{p,r,e}\ge 0 ,\qquad \forall p\in P,\;r\in R,\;e\in E
\]

---

**Data Mapping**

| Parameter | Table ID | Exact Column Name |
|-----------|----------|-------------------|
| \(t_{I,r,e}\) | file_0_view_0 | `Product I` |
| \(t_{II,r,e}\) | file_0_view_0 | `Product II` |
| \(t_{III,r,e}\) | file_0_view_0 | `Product III` |
| \(T_e\) | file_0_view_0 | `Available Equipment Operating Time` |
| \(c^E_e\) | file_0_view_0 | `Equipment Cost at Full Load (yuan)` |
| \(c^{RM}_p\) | file_0_view_0 | *(not present in retrieved data)* |
| \(p_p\) | file_0_view_0 | *(not present in retrieved data)* |

*(All column names are taken verbatim from the retrieved CSV.)*