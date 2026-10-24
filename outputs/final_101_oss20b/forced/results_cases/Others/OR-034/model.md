## Mathematical Model

**Indices**  
\(I\) – set of all baked‑good products (product names from *Frenchbakerydailysales.csv*).

**Parameters**  
\(R_i\) – revenue per unit of product \(i\) (column **Revenue**).  
\(D_i\) – demand for product \(i\) (column **Demand**).  
\(I_i\) – initial inventory of product \(i\) (column **Initial Inventory**).

**Decision Variables**  
\(x_i \ge 0\) – quantity of product \(i\) that the bakery plans to sell.

**Objective**  

\[
\max \;\sum_{i\in I} R_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le D_i &&\forall i\in I &&\text{(demand limit)}\\[4pt]
x_i &\le I_i &&\forall i\in I &&\text{(inventory limit)}\\[4pt]
x_i &\ge 0 &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping Rules

| Table ID | Column | Parameter | Description |
|----------|--------|-----------|-------------|
| `file_0_view_0` | `Product Name` | \(i\) | Index set of products |
| `file_0_view_0` | `Revenue` | \(R_i\) | Revenue per unit of product \(i\) |
| `file_0_view_0` | `Demand` | \(D_i\) | Demand quantity for product \(i\) |
| `file_0_view_0` | `Initial Inventory` | \(I_i\) | Initial inventory available for product \(i\) |

All parameters are taken directly from the corresponding columns of the single returned table; no additional filtering or aggregation is applied. The model uses the full set of 148 products present in the table.