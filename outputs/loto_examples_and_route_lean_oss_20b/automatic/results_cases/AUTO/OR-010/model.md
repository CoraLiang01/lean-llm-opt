**Mathematical Model**

Let  

- \(I\) be the set of products (indexed by the values in the column **Product Name**).  
- \(x_i\) be the number of units of product \(i\) that are fulfilled.  

Parameters (all taken from the table **file_0_view_0**):

- \(\text{rev}_i\) – revenue per unit of product \(i\) (column **Revenue**).  
- \(\text{dem}_i\) – deterministic demand for product \(i\) (column **Demand**).  
- \(\text{inv}_i\) – initial inventory available for product \(i\) (column **Initial Inventory**).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{rev}_i \, x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; \text{dem}_i \qquad \forall i\in I \\[2pt]
& x_i \;\le\; \text{inv}_i \qquad \forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Table ID | Column Name |
|--------|----------|-------------|
| \(I\) | file_0_view_0 | Product Name |
| \(\text{rev}_i\) | file_0_view_0 | Revenue |
| \(\text{dem}_i\) | file_0_view_0 | Demand |
| \(\text{inv}_i\) | file_0_view_0 | Initial Inventory |

All parameters are numeric; the index set \(I\) is derived from the distinct values in the **Product Name** column.