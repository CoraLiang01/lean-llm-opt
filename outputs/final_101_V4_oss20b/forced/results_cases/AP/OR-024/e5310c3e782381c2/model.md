**Mathematical Model**

Let  

- \(I\) – set of products with names beginning with “S700_”.  
- \(r_i\) – revenue per unit of product \(i\).  
- \(d_i\) – deterministic demand for product \(i\).  
- \(v_i\) – initial inventory of product \(i\).  
- \(x_i\) – number of units of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le \min\{d_i,\;v_i\} && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Product names | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – | – |

All parameters are taken directly from the filtered rows of `SampleSalesData.csv` where the `Product Name` column starts with the prefix `S700_`. No numerical values are enumerated in the model.