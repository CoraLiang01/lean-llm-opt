**Mathematical Model**

Let  

- \(I\) – set of all products in the dataset.  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**).  
  - \(d_i\) – deterministic demand over the horizon (from column **Demand**).  
  - \(I_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i\in \mathbb{Z}_{+}\) – number of units of product \(i\) the retailer chooses to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le I_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{+} &&\forall i\in I
\end{aligned}
\]

(If the retailer is required to sell at least one unit of each product, replace the lower bound \(0\) by \(1\).)

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(I_i\)   | `file_0_view_0` | `Initial Inventory` |
| \(I\)     | `file_0_view_0` | `Product Name` (identifiers) |

No additional data or constraints are introduced beyond those specified in the query.