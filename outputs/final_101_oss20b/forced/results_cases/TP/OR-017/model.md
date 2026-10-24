**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & Z = \sum_{i\in I} r_i\,x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le d_i &&\forall i\in I \quad\text{(demand)}\\
& 0 \le x_i \le s_i &&\forall i\in I \quad\text{(inventory)}\\
& x_i \ge 0 &&\forall i\in I
\end{aligned}
\]

**Index set**

- \(I\): set of all product SKUs that start with the prefix “ZZ” (obtained from the filtered view).

**Parameters**

- \(r_i\): revenue per unit of product \(i\) (from column **Revenue**).
- \(d_i\): demand quantity of product \(i\) (from column **Demand**).
- \(s_i\): initial inventory of product \(i\) (from column **Initial Inventory**).

**Decision variables**

- \(x_i\): number of units of product \(i\) to fulfill (continuous, non‑negative).

---

**Data Mapping Rules**

| Source | Table ID | Column | Target | Description |
|--------|----------|--------|--------|-------------|
| RetailStoreSalesTransactions(ScannerData).csv (filtered view) | `file_0_view_0` | `SKU` | `i` | Index of ZZ product |
| | | `Revenue` | `r_i` | Revenue per unit |
| | | `Demand` | `d_i` | Demand quantity |
| | | `Initial Inventory` | `s_i` | Available inventory |

*The filter `SKU` starts with `"ZZ"` ensures that only ZZ products populate set \(I\). No additional aggregation or transformation is applied to the parameters.*