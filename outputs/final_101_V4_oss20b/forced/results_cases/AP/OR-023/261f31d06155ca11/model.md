**Mathematical Model**

Let  

- \(I\) : set of product references whose `Product_Reference` starts with **ELE‑S**.  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(v_i\) : initial inventory (from column **Initial Inventory**)  
  - \(x_i\) : number of units of product \(i\) to sell (decision variable)

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le v_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Source | Table ID | Column | Parameter / Variable |
|--------|----------|--------|----------------------|
| `SalesStoreoverview.csv` | `file_0_view_0` | `Product_Reference` | index \(i\) (filtered by prefix `ELE-S`) |
| | | `Revenue` | \(r_i\) |
| | | `Demand` | \(d_i\) |
| | | `Initial Inventory` | \(v_i\) |

All parameters are taken directly from the corresponding columns of the filtered rows in the table `file_0_view_0`. No numerical values are enumerated in the model.