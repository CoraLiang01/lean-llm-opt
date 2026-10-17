**Mathematical Model**

Let  

* \(P\) – set of products that contain “27in” in the *Product Name* column.  
* For each \(p\in P\)  
  * \(r_p\) – revenue per unit (from column **Revenue**).  
  * \(d_p\) – demand (from column **Demand**).  
  * \(I_p\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_p \ge 0\) – number of units of product \(p\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{p\in P} r_p\,x_p \\[4pt]
\text{s.t. } & x_p \;\le\; d_p && \forall p\in P \\[2pt]
             & x_p \;\le\; I_p && \forall p\in P \\[2pt]
             & x_p \;\ge\; 0   && \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(P\) | Product identifiers | `file_0_view_0` | `Product Name` |
| \(r_p\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_p\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(I_p\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All values are taken directly from the filtered view `file_0_view_0` that contains only rows where **Product Name** contains the substring “27in”. No additional numerical values or constraints are introduced beyond those specified.