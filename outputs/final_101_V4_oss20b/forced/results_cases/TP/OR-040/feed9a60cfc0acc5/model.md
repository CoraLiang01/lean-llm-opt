**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

**Index sets and parameters**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(I\) | Set of development areas (ProductName) | `file_1_view_0` |
| \(v_i\) | Benefit coefficient for area \(i\) | `file_1_view_0` |
| \(C\) | Overall development capacity | `file_0_view_0` |

**Data Mapping Rules**

| Parameter | Table ID | Column(s) | Mapping |
|-----------|----------|-----------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | scalar value |
| \(I\) | `file_1_view_0` | `ProductName` | index set |
| \(v_i\) | `file_1_view_0` | `Value` | parameter indexed by `ProductName` |

All other data are implicit in the model and are not required for the formulation.