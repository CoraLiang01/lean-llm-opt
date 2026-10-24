**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \;\le\; x_i \;\le\; \text{Demand}_i , && \forall i\in I \\[2pt]
& 0 \;\le\; x_i \;\le\; \text{Initial}_i , && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}, && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | `file_0_view_0` | `Demand` |
| \(\text{Initial}_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`. The decision variables \(x_i\) are non‑negative integers representing the number of units of each 27‑inch product to be sold.