**Mathematical Model**

Let  

* \(I\) – set of 4U products (indexed by the value in the column **“Product Name”**).  
* \(r_i\) – revenue per unit of product \(i\) (column **“Revenue”**).  
* \(d_i\) – demand for product \(i\) (column **“Demand”**).  
* \(inv_i\) – initial inventory of product \(i\) (column **“Initial Inventory”**).  
* \(x_i\) – number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.