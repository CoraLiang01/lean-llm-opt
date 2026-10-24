**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le \text{Demand}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{Inventory}_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

**Index Set**

- \(I\): set of tablet product identifiers (taken from the *Product Name* column).

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{Revenue}_i\) | Revenue per unit of product \(i\) | `Revenue` column |
| \(\text{Demand}_i\) | Maximum demand for product \(i\) | `Demand` column |
| \(\text{Inventory}_i\) | Initial inventory of product \(i\) | `Initial Inventory` column |

**Decision Variables**

- \(x_i\): number of units of product \(i\) to fulfill (integer, non‑negative).

---

**Data Mapping Rules**

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Product Name` | Index set \(I\) (identifier for each product) |
| `file_0_view_0` | `Revenue` | Parameter \(\text{Revenue}_i\) |
| `file_0_view_0` | `Demand` | Parameter \(\text{Demand}_i\) |
| `file_0_view_0` | `Initial Inventory` | Parameter \(\text{Inventory}_i\) |

These mappings bind each row of the CSV to the corresponding model element without enumerating individual values. The model uses the supplied data to compute the optimal fulfillment plan that maximizes total revenue while respecting demand and inventory limits.