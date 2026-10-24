**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \;\le\; x_i \;\le\; \text{dem}_i && \forall i\in I \\[2pt]
& 0 \;\le\; x_i \;\le\; \text{inv}_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

*Sets*  
\(I\) – set of product references whose `Product_Reference` starts with **ELE‑S**.

*Parameters*  
\(\text{rev}_i\) – revenue per unit of product \(i\).  
\(\text{dem}_i\) – deterministic demand for product \(i\).  
\(\text{inv}_i\) – initial inventory available for product \(i\).

*Decision Variables*  
\(x_i\) – number of units of product \(i\) to fulfill.

---

**Data Mapping**

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Product_Reference` | index set \(I\) |
| `file_0_view_0` | `Revenue` | parameter \(\text{rev}_i\) |
| `file_0_view_0` | `Demand` | parameter \(\text{dem}_i\) |
| `file_0_view_0` | `Initial Inventory` | parameter \(\text{inv}_i\) |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or assumptions are introduced beyond what is supplied.