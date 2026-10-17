**Mathematical Model**

Let  

* \(W=\{w_1,\dots ,w_{10}\}\) be the set of warehouses (indices taken from the *Warehouse ID* column).  
* \(P=\{p_1,\dots ,p_{10}\}\) be the set of vehicle types (indices taken from the *ProductName* column).  

Parameters  

* \(c_w\) – capacity of warehouse \(w\) (units of weight).  
* \(v_p\) – benefit (value) of one unit of product \(p\).  
* \(w_p\) – weight of one unit of product \(p\).  

Decision variables  

\[
x_{p,w}\in \mathbb{Z}_{+}\quad\text{for all }p\in P,\;w\in W
\]

---

### Objective
\[
\max \;\sum_{p\in P}\sum_{w\in W} v_p\,x_{p,w}
\]

### Constraints
1. **Warehouse capacity**  
   \[
   \sum_{p\in P} w_p\,x_{p,w}\;\le\; c_w \quad\forall\,w\in W
   \]

2. **Non‑negativity & integrality**  
   \[
   x_{p,w}\;\ge\;0,\qquad x_{p,w}\in\mathbb{Z}\quad\forall\,p\in P,\;w\in W
   \]

---

**Data Mapping Rules**

| Parameter | Source Table | Table ID | Index Column | Value Column |
|-----------|--------------|----------|--------------|--------------|
| \(c_w\)   | capacity.csv | file_0_view_0 | Warehouse ID | Capacity |
| \(v_p\)   | products.csv | file_1_view_0 | ProductName | Value |
| \(w_p\)   | products.csv | file_1_view_0 | ProductName | Weight |

*All indices \(w\) and \(p\) are taken directly from the corresponding columns in the source tables; no additional aggregation or transformation is applied.*