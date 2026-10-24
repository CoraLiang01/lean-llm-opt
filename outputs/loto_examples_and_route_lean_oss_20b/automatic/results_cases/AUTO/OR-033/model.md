**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{Demand}_i && \forall i\in I \\[2pt]
& x_i \;\le\; \text{InitInv}_i && \forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | *row identifier* |
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | `file_0_view_0` | `Demand` |
| \(\text{InitInv}_i\) | `file_0_view_0` | `Initial Inventory` |

*All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`. The index set \(I\) consists of all rows returned by that view.*