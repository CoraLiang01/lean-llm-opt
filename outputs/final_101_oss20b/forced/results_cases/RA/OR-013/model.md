**Mathematical Model**

\[
\begin{aligned}
\text{Maximize } & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to } & x_i \;\le\; \text{Demand}_i && \forall i\in I \\[2pt]
& x_i \;\le\; \text{Inv}_i && \forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of products (identified by *Product Name*).

**Parameters**

- \(\text{Revenue}_i\): revenue per unit of product \(i\).
- \(\text{Demand}_i\): known demand for product \(i\).
- \(\text{Inv}_i\): initial inventory available for product \(i\).

**Decision Variables**

- \(x_i\): number of units of product \(i\) to fulfill (integer, \(\ge 0\)).

---

### Data Mapping

| Parameter / Set | Table ID          | Column Name          |
|-----------------|-------------------|----------------------|
| \(I\)           | file_0_view_0     | Product Name         |
| \(\text{Revenue}_i\) | file_0_view_0 | Revenue              |
| \(\text{Demand}_i\)  | file_0_view_0 | Demand               |
| \(\text{Inv}_i\)     | file_0_view_0 | Initial Inventory    |

All parameter values are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or assumptions are introduced.