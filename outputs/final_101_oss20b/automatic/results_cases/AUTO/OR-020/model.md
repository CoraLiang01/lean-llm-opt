**Mathematical Model**

Let  

* \(I\) be the set of all products (indexed by the *Product Name* column).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand (from column **Demand**)  
  * \(s_i\) – initial inventory (from column **Initial Inventory**)  

Decision variable  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le s_i &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

All indices \(i\) are taken from the *Product Name* column of the same table. No additional data or constraints are introduced beyond those explicitly required by the query.