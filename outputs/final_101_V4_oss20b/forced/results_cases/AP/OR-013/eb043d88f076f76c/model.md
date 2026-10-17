**Mathematical Model**

Let  

- \(I\) : set of “4U” products (indexed by product name).  
- \(r_i\) : revenue per unit of product \(i\).  
- \(d_i\) : demand of product \(i\).  
- \(inv_i\) : initial inventory of product \(i\).  
- \(x_i\) : units of product \(i\) to fulfill.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i && \forall i\in I \\[2pt]
             & x_i \;\le\; inv_i && \forall i\in I \\[2pt]
             & x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(r_i\) | `file_0_view_0` | `Revenue` column |
| \(d_i\) | `file_0_view_0` | `Demand` column |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` column |
| \(I\) | `file_0_view_0` | set of distinct `Product Name` values that contain “4U” |

All values are taken directly from the filtered rows of the CSV file; no numerical values are enumerated in the model.