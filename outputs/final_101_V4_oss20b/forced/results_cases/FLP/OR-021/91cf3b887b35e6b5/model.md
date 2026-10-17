**Mathematical Model**

Let  

* \(I\) – set of all products (identified by their *Product Name*).  
* \(rev_i\) – revenue per unit of product \(i\) (from column **Revenue**).  
* \(dem_i\) – demand quantity of product \(i\) (from column **Demand**).  
* \(inv_i\) – initial inventory of product \(i\) (from column **Initial Inventory**).  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) that will be sold (fulfilled).

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} rev_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le dem_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le inv_i && \forall i\in I
\end{aligned}
\]

The two upper‑bound constraints guarantee that sales do not exceed either the
known demand or the available inventory for each product.

---

**Data Mapping**

| Table ID | Column in CSV | Parameter / Variable |
|----------|---------------|----------------------|
| `file_0_view_0` | `Product Name` | index \(i\) (product identifier) |
| `file_0_view_0` | `Revenue` | \(rev_i\) |
| `file_0_view_0` | `Demand` | \(dem_i\) |
| `file_0_view_0` | `Initial Inventory` | \(inv_i\) |

All rows of the table are used to populate the sets and parameters; no
subsetting or filtering is applied. The variable domain is non‑negative
continuous as specified.