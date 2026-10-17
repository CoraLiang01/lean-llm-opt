**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le \text{Demand}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{Inv}_i && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_+ && \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set of Baby products | `file_0_view_0` | *implicit* (row identifiers) |
| \(\text{Revenue}_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(\text{Inv}_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – | – |

*All parameters are taken directly from the returned rows of the filtered view `file_0_view_0`.*