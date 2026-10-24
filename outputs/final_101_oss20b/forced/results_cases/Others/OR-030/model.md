**Mathematical Model**

Let  

\[
I=\{1,\dots,8\}
\]

be the set of the 8 *FDK57* car‑model records that satisfy the filter  
(`Product Name` starts with “FDK57”) in the file **BigMartSales.csv**.

Parameters (indexed by \(i\in I\))  

\[
\begin{aligned}
\text{rev}_i &\;=\; \text{Revenue of record }i \\
\text{dem}_i &\;=\; \text{Demand of record }i \\
\text{inv}_i &\;=\; \text{Initial Inventory of record }i
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{+}\qquad (i\in I)
\]

---

**Objective**

\[
\max \;\sum_{i\in I} \text{rev}_i\,x_i
\]

---

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; \text{dem}_i &&\forall i\in I \\
x_i &\;\le\; \text{inv}_i &&\forall i\in I \\
x_i &\;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Filter |
|-----------|--------------|--------|--------|
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` | `Product Name` starts with `"FDK57"` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` | same filter |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` | same filter |

| Index Set | Source Table | Rows | Filter |
|-----------|--------------|------|--------|
| \(I\) | `file_0_view_0` | 8 rows (source rows 1163, 1501, 1576, 1793, 2438, 4297, 4806, 4942) | `Product Name` starts with `"FDK57"` |

All parameters and variables are defined only for the rows that satisfy the filter; no additional rows are introduced. The model uses integer non‑negative decision variables and respects the per‑record demand and inventory limits.