**Mathematical Model**

Let  

* \(I=\{1,\dots,68\}\) be the set of product records.  
* \(x_i\) – number of units of product \(i\) to be sold (decision variable).  

Parameters (all taken directly from the CSV view `file_0_view_0`):

| Parameter | Source column | Meaning |
|-----------|---------------|---------|
| \(r_i\) | `Revenue` | revenue per unit of product \(i\) |
| \(d_i\) | `Demand` | maximum demand (units) for product \(i\) |
| \(s_i\) | `Initial Inventory` | available inventory for product \(i\) |

All variables are non‑negative integers.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Index | Parameter | Table ID | Column |
|-------|-----------|----------|--------|
| \(i\) | – | `file_0_view_0` | – (row index) |
| \(r_i\) | revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | demand | `file_0_view_0` | `Demand` |
| \(s_i\) | initial inventory | `file_0_view_0` | `Initial Inventory` |

All values are taken as they appear in the CSV; no additional processing or aggregation is performed. The model uses the full set of 68 product records returned by the query.