## Mathematical Model

**Indices**  
\(i \in \mathcal{I}\) – set of products (one element per row of the table).

**Parameters**  
\(R_i\) – revenue per unit of product \(i\).  
\(D_i\) – demand quantity of product \(i\).  
\(I_i\) – initial inventory of product \(i\).

**Decision Variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to be fulfilled.

**Objective**  
\[
\max \;\sum_{i\in\mathcal{I}} R_i\,x_i
\]

**Constraints**  
\[
\begin{aligned}
x_i &\le D_i &&\forall i\in\mathcal{I} \quad\text{(cannot exceed demand)}\\
x_i &\le I_i &&\forall i\in\mathcal{I} \quad\text{(cannot exceed available inventory)}\\
x_i &\ge 0 &&\forall i\in\mathcal{I}
\end{aligned}
\]

---

## Data Mapping

| Parameter | Source Table | Column Name | Notes |
|-----------|--------------|-------------|-------|
| \(R_i\) | `file_0_view_0` | `Revenue` | Revenue per unit |
| \(D_i\) | `file_0_view_0` | `Demand` | Known demand quantity |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` | Initial stock available |

The index set \(\mathcal{I}\) consists of all rows returned in `file_0_view_0` (24 products). No additional filtering or aggregation is applied.