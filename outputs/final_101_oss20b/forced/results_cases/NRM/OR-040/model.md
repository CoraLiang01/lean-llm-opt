**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} b_i\,x_i \\[4pt]
\text{subject to}\quad 
& \sum_{i\in I} x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{+}\qquad \forall i\in I
\end{aligned}
\]

**Index set**

- \(I\): set of all area names (e.g., Queens, Brooklyn, …) appearing in *products.csv*.

**Parameters**

- \(b_i\): benefit coefficient for area \(i\).
- \(C\): overall development capacity.

**Variables**

- \(x_i\): integer scale of development in area \(i\) per day.

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) | Mapping Rule |
|-----------|--------|----------|-----------|--------------|
| \(C\) | *capacity.csv* | `file_0_view_0` | `Capacity` | \(C = \text{Capacity}\) (single row) |
| \(b_i\) | *products.csv* | `file_1_view_0` | `ProductName`, `Value` | For each row, set \(b_{\text{ProductName}} = \text{Value}\) |
| \(I\) | *products.csv* | `file_1_view_0` | `ProductName` | \(I = \{\text{ProductName}\}\) (all distinct values) |

All parameter values are taken directly from the corresponding CSV files; no additional data or assumptions are introduced.