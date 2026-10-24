**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Value}_i \; x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} \text{Weight}_i \; x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Symbol | Source Table | Column |
|--------|---------------|--------|
| \(I\) (index set) | `file_1_view_0` | `ProductName` |
| \(\text{Value}_i\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_i\) | `file_1_view_0` | `Weight` |
| \(\text{Capacity}\) | `file_0_view_0` | `Capacity` |

*All parameters are taken directly from the corresponding CSV views; no values are enumerated or altered.*