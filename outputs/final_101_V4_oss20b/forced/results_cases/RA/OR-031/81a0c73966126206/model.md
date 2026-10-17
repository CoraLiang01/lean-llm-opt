**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le \text{Demand}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{InitInv}_i && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

### Data Mapping

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Product index | `file_0_view_0` | `Full_Product_Name` |
| \(\text{Revenue}_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(\text{InitInv}_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the single table `file_0_view_0`. No additional data or assumptions are introduced.