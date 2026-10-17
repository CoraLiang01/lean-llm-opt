**Mathematical Model**

Let  

* \(I=\{i\mid\text{Product Name}_i\text{ starts with }S700_\}\)  
* \(r_i\) – revenue per unit of product \(i\) (from column **Revenue**)  
* \(d_i\) – demand quantity of product \(i\) (from column **Demand**)  
* \(v_i\) – initial inventory of product \(i\) (from column **Initial Inventory**)  
* \(x_i\) – number of units of product \(i\) to sell (decision variable)

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le \min\{d_i,\;v_i\}\qquad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{+}\qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Source | Filter |
|----------|--------|--------|--------|
| `file_0_view_0` | `Product Name` | `SampleSalesData.csv` | `Product Name` starts with `S700_` |
| `file_0_view_0` | `Revenue` | `SampleSalesData.csv` | – |
| `file_0_view_0` | `Demand` | `SampleSalesData.csv` | – |
| `file_0_view_0` | `Initial Inventory` | `SampleSalesData.csv` | – |

*Index set* \(I\) is defined by the filter on `Product Name`.  
*Parameters* \(r_i, d_i, v_i\) are taken directly from the corresponding columns of the filtered rows.  
*Decision variables* \(x_i\) are non‑negative integers bounded by the minimum of demand and available inventory.