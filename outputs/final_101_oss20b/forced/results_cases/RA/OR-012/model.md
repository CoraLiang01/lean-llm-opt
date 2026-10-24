**Mathematical Model**

Let  

* \(I\) be the set of all products (identified by the *Product Name* column).  
* For each \(i\in I\)  
  * \(rev_i\) – revenue per unit (column **Revenue**)  
  * \(dem_i\) – deterministic demand (column **Demand**)  
  * \(inv_i\) – initial inventory (column **Initial Inventory**)  

Decision variable  

* \(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;&\sum_{i\in I} rev_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le dem_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le inv_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|---------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(rev_i\) | `file_0_view_0` | `Revenue` |
| \(dem_i\) | `file_0_view_0` | `Demand` |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`; no values are enumerated or altered.