**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \;\le\; x_i \;\le\; \text{Demand}_i && \forall i\in I \\[2pt]
& 0 \;\le\; x_i \;\le\; \text{InitInv}_i && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

*Indices*  
\(I\) – set of all products in the dataset.

*Parameters*  
\(\text{Revenue}_i\) – revenue per unit of product \(i\).  
\(\text{Demand}_i\) – deterministic demand for product \(i\).  
\(\text{InitInv}_i\) – initial inventory available for product \(i\).

*Decision Variables*  
\(x_i\) – number of units of product \(i\) to sell (integer, non‑negative).

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `"Product Name"` |
| \(\text{Revenue}_i\) | `file_0_view_0` | `"Revenue"` |
| \(\text{Demand}_i\) | `file_0_view_0` | `"Demand"` |
| \(\text{InitInv}_i\) | `file_0_view_0` | `"Initial Inventory"` |

All parameters and the index set are obtained directly from the rows of `file_0_view_0`; no additional data or assumptions are introduced.