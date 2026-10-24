**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_1_view_0` | `ProductName` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(C\) | `file_0_view_0` | `Capacity` |

All parameters are taken directly from the corresponding CSV views; no values are enumerated in the model.