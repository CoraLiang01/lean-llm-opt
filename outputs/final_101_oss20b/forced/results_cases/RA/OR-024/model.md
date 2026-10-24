**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le \text{dem}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{inv}_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | *index set* | product name (`Product Name`) |
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

*All parameters are taken directly from the rows returned by the filter on `Product Name` with prefix `S700_`. No additional data or assumptions are introduced.*