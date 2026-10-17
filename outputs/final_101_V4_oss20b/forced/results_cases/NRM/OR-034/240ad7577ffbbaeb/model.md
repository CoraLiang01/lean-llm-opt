**Symbolic Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le I_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_+ &&\forall i\in I
\end{aligned}
\]

**Index Sets**

| Symbol | Description |
|--------|-------------|
| \(I\)  | Set of all baked‑good products in the dataset |

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(r_i\) | Revenue per unit of product \(i\) | `Revenue` column |
| \(d_i\) | Total demand for product \(i\) | `Demand` column |
| \(I_i\) | Initial inventory of product \(i\) | `Initial Inventory` column |

**Variables**

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(x_i\) | \(\mathbb{Z}_+\) (non‑negative integer) | Quantity of product \(i\) the bakery plans to sell |

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are mapped directly to the corresponding columns of the single source table `file_0_view_0`. No additional data or constraints are introduced beyond those specified in the query.