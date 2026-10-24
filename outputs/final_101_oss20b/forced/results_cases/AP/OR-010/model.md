**Mathematical Model**

Let  

- \(I\) : set of products (indexed by product name).  
- \(r_i\) : revenue per unit of product \(i\).  
- \(d_i\) : deterministic demand for product \(i\).  
- \(inv_i\) : initial inventory of product \(i\).  
- \(x_i\) : integer quantity of product \(i\) to fulfill.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Parameter |
|----------|--------|-----------|
| file_0_view_0 | Product Name | index set \(I\) |
| file_0_view_0 | Revenue | \(r_i\) |
| file_0_view_0 | Demand | \(d_i\) |
| file_0_view_0 | Initial Inventory | \(inv_i\) |

All parameter values are taken directly from the corresponding columns of the CSV file `MobileSalesDataset.csv`. No additional data or enumeration is required.