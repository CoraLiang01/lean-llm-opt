**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \;\le\; x_i \;\le\; \text{Demand}_i , && \forall i\in I \\[2pt]
& 0 \;\le\; x_i \;\le\; \text{Inv}_i , && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}, && \forall i\in I
\end{aligned}
\]

*Indices*  
\(I\) – set of all products (identified by their product name).

*Parameters*  
\(\text{Revenue}_i\) – revenue per unit of product \(i\).  
\(\text{Demand}_i\) – maximum demand for product \(i\).  
\(\text{Inv}_i\) – initial inventory available for product \(i\).

*Decision variables*  
\(x_i\) – number of units of product \(i\) to sell (integer, non‑negative).

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `"Product Name"` |
| \(\text{Revenue}_i\) | `file_0_view_0` | `"Revenue"` |
| \(\text{Demand}_i\) | `file_0_view_0` | `"Demand"` |
| \(\text{Inv}_i\) | `file_0_view_0` | `"Initial Inventory"` |

All parameters are taken directly from the corresponding columns of the single table `file_0_view_0`; no additional data or assumptions are introduced.