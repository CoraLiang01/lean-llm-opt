**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \; \text{revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \;\le\; x_i \;\le\; \text{demand}_i , \qquad \forall i\in I \\[2pt]
& 0 \;\le\; x_i \;\le\; \text{inventory}_i , \qquad \forall i\in I \\[2pt]
& x_i \;\ge\; 0 , \qquad \forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of all products whose *Product Name* contains the substring “27in” (obtained by the filter in the query).

**Parameters**

- \(\text{revenue}_i\): revenue per unit for product \(i\) (from column *Revenue*).
- \(\text{demand}_i\): deterministic demand quantity for product \(i\) (from column *Demand*).
- \(\text{inventory}_i\): initial inventory level for product \(i\) (from column *Initial Inventory*).

**Decision Variables**

- \(x_i \ge 0\): number of units of product \(i\) to fulfill (continuous).

---

### Data Mapping Rules

| Parameter / Variable | Source Table | Source Column |
|----------------------|--------------|---------------|
| \(I\) (index set)    | `file_0_view_0` | `Product Name` (filtered by `contains "27in"`) |
| \(\text{revenue}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{demand}_i\)  | `file_0_view_0` | `Demand` |
| \(\text{inventory}_i\) | `file_0_view_0` | `Initial Inventory` |
| \(x_i\)              | —            | — |

*All mappings use the exact column names as they appear in the source CSV. No additional data or aggregation is introduced.*