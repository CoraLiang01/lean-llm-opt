**Mathematical Model**

Let  

- \(P\) be the set of products.  
- For each \(p\in P\)  
  - \(r_p\) : revenue per unit (dollar).  
  - \(d_p\) : demand quantity (units).  
  - \(i_p\) : initial inventory (units).  
- Decision variable  
  - \(x_p\ge 0\) : quantity of product \(p\) to sell.

\[
\begin{aligned}
\max\;& \sum_{p\in P} r_p\,x_p \\[4pt]
\text{s.t.}\;& 0 \le x_p \le d_p &&\forall p\in P \\[2pt]
& 0 \le x_p \le i_p &&\forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(P\) | set of product names | `file_0_view_0` | `Product Name` |
| \(r_p\) | revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_p\) | demand quantity | `file_0_view_0` | `Demand` |
| \(i_p\) | initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the CSV file `RetailSalesDataset.csv` (view `file_0_view_0`). No additional constraints or parameters are introduced beyond those required by the problem statement.