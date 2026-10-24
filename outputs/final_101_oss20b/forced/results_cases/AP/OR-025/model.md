**Mathematical Model**

Let  

- \(I\) – set of tablet models extracted from the filtered table.  
- \(r_i\) – revenue per unit of model \(i\).  
- \(d_i\) – demand quantity of model \(i\).  
- \(inv_i\) – initial inventory of model \(i\).  
- \(x_i\) – number of units of model \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le \min\{d_i,\;inv_i\}, && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}, && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | set of rows returned by filter | `file_0_view_0` | `Product Name` |
| \(r_i\) | revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | demand quantity | `file_0_view_0` | `Demand` |
| \(inv_i\) | initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the filtered rows in the CSV file `SmartphoneRetailOutletSalesData.csv`. No numerical values are enumerated in the model.