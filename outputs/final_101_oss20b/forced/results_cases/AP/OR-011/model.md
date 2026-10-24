**Mathematical Model**

Let  

- \(I\) be the set of products that satisfy the filter \(\text{prefix}(id\_number)=\text{“id999”}\).  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand during the horizon (from column **Demand**)  
  - \(inv_i\) : initial inventory (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i\in\mathbb{Z}_{+}\) : units of product \(i\) to fulfill  

The model is

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; inv_i && \forall i\in I \\[2pt]
             & x_i \;\le\; d_i   && \forall i\in I \\[2pt]
             & x_i \;\in\;\mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_i\)   | file_0_view_0 | Revenue |
| \(d_i\)   | file_0_view_0 | Demand |
| \(inv_i\) | file_0_view_0 | Initial Inventory |

| Variable | Domain |
|----------|--------|
| \(x_i\)  | integer, \(x_i \ge 0\) |

*Table ID:* `file_0_view_0` (filtered view of `OnlineRetailSalesDataset.csv` where `id_number` starts with `id999`).  
*Returned rows:* 1 (product `id999`).