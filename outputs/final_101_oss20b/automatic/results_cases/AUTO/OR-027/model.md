**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{d}_i &&\forall i\in I \quad\text{(demand limit)}\\
& x_i \;\le\; \text{inv}_i &&\forall i\in I \quad\text{(inventory limit)}\\[4pt]
& x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

**Index set**

- \(I\): set of all products whose *Sub Category* contains the string “Organ” in the source file.

**Parameters**

- \(\text{rev}_i\): revenue per unit of product \(i\).
- \(\text{d}_i\): total demand for product \(i\) over the sales horizon.
- \(\text{inv}_i\): initial inventory available for product \(i\).

**Variables**

- \(x_i\): number of units of product \(i\) to be fulfilled (integer, non‑negative).

---

### Data Mapping

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{d}_i\)   | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

*Note:* The index set \(I\) is derived from the rows of `file_0_view_0` that satisfy the filter condition `Sub Category` contains `"Organ"`. No other data or constraints are introduced beyond those specified in the query.