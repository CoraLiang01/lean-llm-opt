**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{dem}_i && \forall i\in I \\[2pt]
& x_i \;\le\; \text{inv}_i && \forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column Name |
|--------|--------------|-------------|
| \(I\) | `file_0_view_0` | `Product_Name` |
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the rows returned in the observation. The decision variables \(x_i\) represent the number of units of each book product to fulfill.