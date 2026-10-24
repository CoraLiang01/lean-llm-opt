**Mathematical Model**

Let  

* \(I\) be the set of all products whose name starts with “Books” (the 5 rows returned by the filter).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (column **Revenue**)  
  * \(d_i\) – demand (column **Demand**)  
  * \(s_i\) – initial inventory (column **Initial Inventory**)  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) that will be sold (continuous).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; s_i &&\forall i\in I \\[2pt]
             & x_i \;\ge\; 0   &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter / Variable | Source Table | Source Column | Notes |
|----------------------|--------------|---------------|-------|
| \(I\) | `file_0_view_0` | `Product_Name` | Set of rows after the “Books” prefix filter |
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit for product \(i\) |
| \(d_i\) | `file_0_view_0` | `Demand` | Demand quantity for product \(i\) |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | Initial inventory for product \(i\) |
| \(x_i\) | – | – | Decision variable, continuous, non‑negative |

All indices, parameters, and constraints are defined solely from the filtered rows of the supplied CSV file. No additional data or assumptions are introduced.