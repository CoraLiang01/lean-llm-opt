**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* contains the string **“27in”** (obtained from the filtered view `file_0_view_0`).  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (parameter)  
  - \(d_i\)  – demand quantity (parameter)  
  - \(s_i\)  – initial inventory (parameter)  

Decision variable  

- \(x_i\) – number of units of product \(i\) to be sold (integer, non‑negative).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le s_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_+ && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\)   | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\)   | Demand | `file_0_view_0` | `Demand` |
| \(s_i\)   | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | Product Name | `file_0_view_0` | `Product Name` |

The index set \(I\) is defined as the set of all distinct values in the *Product Name* column that satisfy the filter condition `"contains" "27in"`. No other tables or columns are referenced.