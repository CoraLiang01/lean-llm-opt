## Mathematical Model

**Index set**

\[
I=\{i\mid i \text{ is a product in the dataset}\}
\]

**Parameters**

\[
\begin{aligned}
\text{Revenue}_i &\;=\; \text{value of column “Revenue” for product }i,\\
\text{Demand}_i   &\;=\; \text{value of column “Demand” for product }i,\\
\text{Inv}_i      &\;=\; \text{value of column “Initial Inventory” for product }i.
\end{aligned}
\]

**Decision variables**

\[
x_i \;\;=\;\; \text{number of units of product }i \text{ that are fulfilled},\qquad
x_i \in \mathbb{Z}_{+}\;\; \forall i\in I.
\]

**Objective**

\[
\max \;\sum_{i\in I}\text{Revenue}_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le \text{Demand}_i &&\forall i\in I,\\
x_i &\le \text{Inv}_i     &&\forall i\in I,\\
x_i &\ge 0                &&\forall i\in I.
\end{aligned}
\]

---

## Data Mapping

| Index | Table ID | Column | Mapping |
|-------|----------|--------|---------|
| \(i\) | `file_0_view_0` | `Full_Product_Name` | Defines the product identifier for each row. |
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` | Numeric revenue per unit for product \(i\). |
| \(\text{Demand}_i\) | `file_0_view_0` | `Demand` | Maximum demand quantity for product \(i\). |
| \(\text{Inv}_i\) | `file_0_view_0` | `Initial Inventory` | Initial inventory available for product \(i\). |

No additional data or transformations are required beyond the direct column values. The model uses all 40 products present in the dataset.