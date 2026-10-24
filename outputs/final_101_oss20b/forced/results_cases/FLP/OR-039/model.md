**Mathematical Model**

Let  

* \(I\) – set of product types (indexed by the *ProductName* values).  
* \(W\) – set of warehouses (indexed by the *Warehouse ID* values).  

Parameters  

* \(v_i\) – benefit (value) of one unit of product \(i\) (from *products.csv*).  
* \(C_w\) – capacity of warehouse \(w\) (from *capacity.csv*).  

Decision variables  

* \(x_{i,w}\in \mathbb{Z}_{+}\) – number of units of product \(i\) stored in warehouse \(w\).  

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{w\in W} v_i\,x_{i,w} \\[4pt]
\text{s.t.}\;& \sum_{i\in I} x_{i,w} \;\le\; C_w , && \forall w\in W \\[4pt]
& x_{i,w}\;\ge\;0 , && \forall i\in I,\; w\in W \\[4pt]
& x_{i,w}\;\in\;\mathbb{Z}, && \forall i\in I,\; w\in W .
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Parameter | Source Table | Column | Mapping |
|-----------|--------------|--------|---------|
| \(v_i\) | `file_1_view_0` | `Value` | \(v_{\text{ProductName}} = \text{Value}\) |
| \(C_w\) | `file_0_view_0` | `Capacity` | \(C_{\text{Warehouse ID}} = \text{Capacity}\) |

*Index sets*  
- \(I = \{\text{ProductName values from `file_1_view_0`}\}\)  
- \(W = \{\text{Warehouse ID values from `file_0_view_0`}\}\)

No other parameters or constraints are introduced. The weight column in *products.csv* is not used in this formulation.