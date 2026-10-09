Symbolic Mathematical Model

Index Sets:
Let $I$ be the set of all products classified under ‘Fashion’ in table_id file_0_view_0, column "Product Name".

Parameters:
For each $i \in I$:
- $r_i$: revenue per unit of product $i$ (from "Revenue")
- $d_i$: deterministic demand for product $i$ (from "Demand")
- $s_i$: initial inventory for product $i$ (from "Initial Inventory")

Decision Variables:
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \\
& x_i \leq s_i && \forall i \in I \\
& x_i \geq 0 && \forall i \in I,\ x_i \in \mathbb{Z}
\end{align*}
\]

Data Mapping:
- Index set $I$ and all parameters $r_i$, $d_i$, $s_i$ are defined by all rows in table_id file_0_view_0, columns "Product Name", "Revenue", "Demand", and "Initial Inventory" of SupermarketSales.csv, filtered to products classified under ‘Fashion’ (prefix "Fashion" in "Product Name").