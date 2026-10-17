**Mathematical Model**

Let  

* \(I\) – set of all products (identified by *Product Name* in the table).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (column *Revenue*).  
  * \(d_i\) – demand (column *Demand*).  
  * \(s_i\) – initial inventory (column *Initial Inventory*).  

Decision variable  

* \(x_i\) – number of units of product \(i\) to sell (integer, \(x_i\ge 0\)).

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;&0 \le x_i \le \min\{d_i,\;s_i\}\qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Columns | Mapping |
|-----------|--------|----------|---------|---------|
| \(r_i\) | Revenue | file_0_view_0 | *Revenue* | \(r_i =\) value in *Revenue* for product \(i\) |
| \(d_i\) | Demand | file_0_view_0 | *Demand* | \(d_i =\) value in *Demand* for product \(i\) |
| \(s_i\) | Initial Inventory | file_0_view_0 | *Initial Inventory* | \(s_i =\) value in *Initial Inventory* for product \(i\) |
| \(I\) | Product set | file_0_view_0 | *Product Name* | \(I =\) set of all *Product Name* values in the table |

No additional constraints or parameters are required beyond those listed.