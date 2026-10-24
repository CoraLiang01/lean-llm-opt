**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & Z = \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{dem}_i &&\forall i\in I \quad\text{(demand)}\\
& x_i \;\le\; \text{inv}_i &&\forall i\in I \quad\text{(inventory)}\\
& x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of FAUX products (rows where *Product Name* contains “FAUX”).

**Parameters**

- \(\text{rev}_i\): revenue per unit of product \(i\).
- \(\text{dem}_i\): demand quantity of product \(i\).
- \(\text{inv}_i\): initial inventory of product \(i\).

**Decision Variables**

- \(x_i\): number of units of product \(i\) to fulfill (continuous, non‑negative).

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Index Mapping |
|-----------|--------------|---------------|---------------|
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` | `i` ← `Product Name` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` | `i` ← `Product Name` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` | `i` ← `Product Name` |

*Only rows where `Product Name` contains the substring “FAUX” are included in set \(I\).*