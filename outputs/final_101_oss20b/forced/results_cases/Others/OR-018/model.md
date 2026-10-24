## Mathematical Model

Let  

- \(I\) be the set of all products whose **Product Name** starts with “Baby” (the rows returned by the filter).  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**).  
  - \(d_i\)  – demand (from column **Demand**).  
  - \(s_i\)  – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i\) – number of units of product \(i\) to sell (integer, \(x_i\ge 0\)).

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;&0 \le x_i \le d_i &&\forall i\in I \\[2pt]
&0 \le x_i \le s_i &&\forall i\in I \\[2pt]
&x_i \in \mathbb{Z}_+ &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping Rules

| Element | Source | Mapping |
|---------|--------|---------|
| **Index set** \(I\) | `file_0_view_0` (Salesdata.csv filtered) | All rows where **Product Name** has prefix “Baby”. |
| **Parameter** \(r_i\) | `file_0_view_0` | Column **Revenue**. |
| **Parameter** \(d_i\) | `file_0_view_0` | Column **Demand**. |
| **Parameter** \(s_i\) | `file_0_view_0` | Column **Initial Inventory**. |
| **Decision variable** \(x_i\) | Decision space | Integer, non‑negative, one per row in \(I\). |

No other parameters or constraints are introduced beyond those explicitly stated in the problem description.