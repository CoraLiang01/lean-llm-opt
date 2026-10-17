**Mathematical Model**

Let  

* \(I\) be the set of FAUX products (identified by their product name).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand (from column **Demand**)  
  * \(inv_i\) – initial inventory (from column **Initial Inventory**)  
  * \(x_i\) – number of units of product \(i\) to be sold  

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_+ && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Column | Table ID | Notes |
|-----------|---------------|----------|-------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue per unit for product \(i\) |
| \(d_i\) | Demand | file_0_view_0 | Deterministic demand for product \(i\) |
| \(inv_i\) | Initial Inventory | file_0_view_0 | Available inventory for product \(i\) |
| \(x_i\) | Decision variable | – | Integer quantity to sell |

*Index set \(I\) is derived from all rows in `file_0_view_0` where **Product Name** contains the substring `"FAUX"`.*