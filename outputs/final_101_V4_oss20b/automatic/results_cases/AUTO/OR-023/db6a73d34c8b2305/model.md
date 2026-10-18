**Abstract Optimization Model**

**Index Set**  
\(I\) – set of all products whose reference starts with “ELE‑S”.

**Parameters**  
\[
\begin{aligned}
w_i &\;=\; \text{Revenue of product } i, && i \in I\\
d_i &\;=\; \text{Demand of product } i, && i \in I\\
c_i &\;=\; \text{Initial inventory of product } i, && i \in I
\end{aligned}
\]

**Decision Variables**  
\[
x_i \;\in\; \mathbb{Z}, \quad x_i \;\ge\; 1, \qquad i \in I
\]

**Objective Function**  
\[
\max \;\sum_{i \in I} w_i \, x_i
\]

**Constraints**

1. **Inventory Constraints**  
\[
x_i \;\le\; c_i, \qquad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \;\le\; d_i, \qquad \forall i \in I
\]

---

### Data Mapping

| Source | Table ID | Column Name | Description |
|--------|----------|-------------|-------------|
| SalesStoreoverview.csv | `file_0_view_0` | `Product_Reference` | Product identifier (prefix “ELE‑S”) |
| SalesStoreoverview.csv | `file_0_view_0` | `Revenue` | Revenue per unit |
| SalesStoreoverview.csv | `file_0_view_0` | `Demand` | Total demand over the sales horizon |
| SalesStoreoverview.csv | `file_0_view_0` | `Initial Inventory` | Initial stock available |

*All parameter values \(w_i, d_i, c_i\) are extracted from the rows of `file_0_view_0` that satisfy the filter condition `Product_Reference` starts with “ELE‑S”.*