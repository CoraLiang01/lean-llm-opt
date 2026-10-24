**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{profit}_i\,x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} \text{weight}_i\,x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Parameter / Set | Source Table | Column |
|-----------------|--------------|--------|
| \(I\) (bread types) | `file_1_view_0` | `ProductName` |
| \(\text{profit}_i\) | `file_1_view_0` | `Value` |
| \(\text{weight}_i\) | `file_1_view_0` | `Weight` |
| \(\text{Capacity}\) | `file_0_view_0` | `Capacity` |

*All numerical values are taken directly from the corresponding CSV files; no additional data are introduced.*