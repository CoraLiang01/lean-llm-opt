**Symbolic Model**

Let  

* \(I\) – set of product references that start with “ELE‑S” (obtained from the table).  
* Parameters  
  * \(r_i\) – revenue per unit of product \(i\).  
  * \(d_i\) – demand quantity of product \(i\).  
  * \(inv_i\) – initial inventory of product \(i\).  

* Decision variable  
  * \(x_i \ge 0\) – number of units of product \(i\) to sell (fulfill).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le inv_i && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | `Product_Reference` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(inv_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All values are taken directly from the filtered rows of `SalesStoreoverview.csv` where `Product_Reference` starts with `ELE‑S`. No additional constraints or parameters are introduced.