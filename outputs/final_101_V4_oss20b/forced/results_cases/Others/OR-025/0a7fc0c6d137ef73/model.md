**Mathematical Model**

Let  

* \(I\) – set of tablet models that satisfy the filter  
  \(\text{Product Name contains “TABLET”}\) (rows 454–480 of *SmartphoneRetailOutletSalesData.csv*).  

Parameters (indexed by \(i\in I\))  

* \(r_i\) – revenue per unit (column **Revenue**)  
* \(d_i\) – demand quantity (column **Demand**)  
* \(v_i\) – initial inventory (column **Initial Inventory**)  

Decision variable  

* \(x_i\) – number of units of model \(i\) to sell (integer, \(x_i\ge 0\))  

Objective  

\[
\max \sum_{i\in I} r_i\,x_i
\]

Constraints  

\[
0 \le x_i \le \min\{d_i,\;v_i\}\qquad \forall i\in I
\]

(Equivalently, two inequalities: \(x_i\le d_i\) and \(x_i\le v_i\).)

---

**Data‑Mapping Rules**

| Index | Source | Column | Mapping |
|-------|--------|--------|---------|
| \(i\) | *file_0_view_0* | *Product Name* | include row if contains “TABLET” |
| \(r_i\) | *file_0_view_0* | **Revenue** | revenue per unit |
| \(d_i\) | *file_0_view_0* | **Demand** | demand quantity |
| \(v_i\) | *file_0_view_0* | **Initial Inventory** | available inventory |

All parameters are taken directly from the corresponding columns of the filtered rows; no additional processing or aggregation is performed. The variable domain is integer non‑negative, consistent with the interpretation of units sold.