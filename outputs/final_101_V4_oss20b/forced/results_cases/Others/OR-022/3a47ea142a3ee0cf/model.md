**Mathematical Model**

Let  

* \(I=\{i\mid\) row \(i\) of the filtered table “Salesorders.csv”\(\}\)  
* \(r_i\) – revenue per unit of product \(i\) (column **Revenue**)  
* \(d_i\) – demand for product \(i\) (column **Demand**)  
* \(s_i\) – initial inventory of product \(i\) (column **Initial Inventory**)  

Decision variable  

* \(x_i\in\mathbb Z_{\ge 0}\) – number of units of product \(i\) that will be sold.

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;&0\le x_i \le d_i &&\forall i\in I \\[2pt]
&0\le x_i \le s_i &&\forall i\in I
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Table | Source | Filter | Index Set | Parameters | Variables |
|-------|--------|--------|-----------|------------|-----------|
| `file_0_view_0` | `Salesorders.csv` | `Product Name` contains `"27in"` | \(I\) | `Revenue` → \(r_i\) <br> `Demand` → \(d_i\) <br> `Initial Inventory` → \(s_i\) | \(x_i\) |

* The filter condition is applied **before** any other processing; only rows satisfying `Product Name` contains `27in` are included in \(I\).  
* All parameters are taken directly from the corresponding columns of the filtered rows.  
* Decision variables \(x_i\) are defined for each \(i\in I\) and are constrained to be non‑negative integers.  

No other data or constraints are introduced beyond those explicitly stated in the query.