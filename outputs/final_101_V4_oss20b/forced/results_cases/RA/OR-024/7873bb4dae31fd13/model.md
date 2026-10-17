**Mathematical Model**

Let  

- \(I\) be the set of products whose name starts with “S700_”.  
- For each \(i\in I\):  
  - \(rev_i\) – revenue per unit (from column **Revenue**).  
  - \(dem_i\) – demand quantity (from column **Demand**).  
  - \(inv_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} rev_i \, x_i \\[4pt]
\text{s.t.}\;& x_i \le dem_i && \forall i\in I \\[2pt]
& x_i \le inv_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(rev_i\) | `file_0_view_0` | `Revenue` |
| \(dem_i\) | `file_0_view_0` | `Demand` |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | – | – |

All indices and parameters are bound directly to the corresponding columns of the table `file_0_view_0`. No additional data or constraints are introduced.