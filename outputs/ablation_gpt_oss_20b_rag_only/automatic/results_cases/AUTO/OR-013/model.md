**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* contains “4U” (the rows returned by the query).  
* For each \(i\in I\)  
  * \(rev_i\) – revenue per unit of product \(i\).  
  * \(dem_i\) – demand for product \(i\).  
  * \(inv_i\) – initial inventory of product \(i\).  
  * \(x_i\) – number of units of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max\;& Z = \sum_{i\in I} rev_i \; x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le dem_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Table ID | Column Name |
|--------|----------|-------------|
| \(I\) (index) | `file_0_view_0` | `Product Name` |
| \(rev_i\) | `file_0_view_0` | `Revenue` |
| \(dem_i\) | `file_0_view_0` | `Demand` |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the returned rows of the CSV file; no additional data or assumptions are introduced.