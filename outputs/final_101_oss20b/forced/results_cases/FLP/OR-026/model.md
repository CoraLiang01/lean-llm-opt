**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* contains the word “Fashion”.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i\) – number of units of product \(i\) that will be sold (fulfilled).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le s_i && \forall i\in I
\end{aligned}
\]

The two upper‑bound constraints guarantee that sales do not exceed the known demand or the available inventory.

---

**Data Mapping**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(I\) | `file_0_view_0` | `Product Name` | All rows where the string “Fashion” appears (case‑sensitive) |
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit for product \(i\) |
| \(d_i\) | `file_0_view_0` | `Demand` | Deterministic demand for product \(i\) |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | Inventory available for product \(i\) |

The mapping uses the raw column names exactly as they appear in the CSV view `file_0_view_0`. No additional transformations or aggregations are applied.