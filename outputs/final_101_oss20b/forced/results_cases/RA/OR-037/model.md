**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Value}_i \; x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} \text{Weight}_i \; x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\end{aligned}
\]

**Index Sets**

* \(I\) – set of product types (identified by `ProductName`).

**Parameters**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(\text{Value}_i\) | Profit per unit of product \(i\) | `file_1_view_0`, column `Value` |
| \(\text{Weight}_i\) | Space/weight per unit of product \(i\) | `file_1_view_0`, column `Weight` |
| \(\text{Capacity}\) | Total inventory capacity | `file_0_view_0`, column `Capacity` |

**Decision Variables**

* \(x_i\) – number of units of product \(i\) to order per day, integer and non‑negative.

---

**Data Mapping Rules**

| Parameter | Table ID | Column | Key |
|-----------|----------|--------|-----|
| `Capacity` | `file_0_view_0` | `Capacity` | – |
| `Value_i` | `file_1_view_0` | `Value` | `ProductName` |
| `Weight_i` | `file_1_view_0` | `Weight` | `ProductName` |

These mappings bind each symbolic parameter to the exact column in the corresponding CSV view, preserving the original identifiers (`ProductName`) and ensuring that the model uses the supplied data without any hard‑coded values.