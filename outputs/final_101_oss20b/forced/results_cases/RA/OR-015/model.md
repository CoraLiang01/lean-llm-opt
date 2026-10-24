**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad 
& 0 \le x_i \le \text{dem}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{init}_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Parameter | Source Table | Source Column | Index Key |
|-----------|--------------|---------------|-----------|
| \(I\) | `file_0_view_0` | `Product Name` | `i` |
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` | `i` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` | `i` |
| \(\text{init}_i\) | `file_0_view_0` | `Initial Inventory` | `i` |

All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.