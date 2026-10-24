**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\end{aligned}
\]

**Index Sets**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(I\)  | Set of produce types (product names) | `file_1_view_0.ProductName` |

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(C\)  | Overall weight capacity | `file_0_view_0.Capacity` |
| \(w_i\) | Weight per unit of product \(i\) | `file_1_view_0.Weight` |
| \(v_i\) | Benefit per unit of product \(i\) | `file_1_view_0.Value` |

**Decision Variables**

| Symbol | Meaning | Domain |
|--------|---------|--------|
| \(x_i\) | Number of units of product \(i\) to order daily | \(\mathbb{Z}_{\ge 0}\) |

---

**Data Mapping Rules**

| Parameter / Variable | Table ID | Column Name |
|----------------------|----------|-------------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(I\) | `file_1_view_0` | `ProductName` |

All values are taken directly from the corresponding CSV views; no enumeration or hard‑coding of individual rows is performed.