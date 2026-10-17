**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & Z = \sum_{i\in I} r_i\,x_i \\[4pt]
\text{subject to}\quad 
& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le s_i &&\forall i\in I \\[4pt]
& x_i \;\text{continuous} &&\forall i\in I
\end{aligned}
\]

*Index set*  
\(I = \{\,i \mid\) product *i* appears in the filtered view `file_0_view_0`\(\}\).

*Parameters*  
\(r_i\) – revenue per unit of product *i* (from column **Revenue**)  
\(d_i\) – demand quantity of product *i* (from column **Demand**)  
\(s_i\) – initial inventory of product *i* (from column **Initial Inventory**)

*Decision variables*  
\(x_i\) – number of units of product *i* to fulfill (continuous, non‑negative)

---

**Data Mapping Rules**

| Table ID | Column Name | Parameter / Variable | Source Column | Notes |
|----------|-------------|----------------------|---------------|-------|
| `file_0_view_0` | `Product Name` | index \(i\) | – | Each distinct product name becomes an index in \(I\). |
| `file_0_view_0` | `Revenue` | \(r_i\) | `Revenue` | Numeric value, unit revenue. |
| `file_0_view_0` | `Demand` | \(d_i\) | `Demand` | Numeric value, deterministic demand. |
| `file_0_view_0` | `Initial Inventory` | \(s_i\) | `Initial Inventory` | Numeric value, available stock. |

*All numeric values are taken directly from the corresponding columns of the filtered view; no aggregation or transformation is performed beyond the type conversion to numeric.*